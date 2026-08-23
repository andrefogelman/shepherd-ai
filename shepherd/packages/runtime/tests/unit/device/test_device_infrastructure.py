"""Unit tests for device selection infrastructure.

Tests the device registry, context manager, and context variable behavior.
"""

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest
from shepherd_core.foundation.protocols.device import DeviceCapabilities
from shepherd_runtime.device import (
    Device,
    _device_registry,
    get_current_device,
    get_device,
    list_devices,
    register_device,
)
from shepherd_runtime.device.local import LocalDevice
from shepherd_runtime.scope import Scope


class TestDeviceRegistry:
    """Tests for device registration and lookup."""

    def test_register_device(self):
        """Test registering a custom device."""
        mock_device = MagicMock()
        mock_device.name = "test-device"

        register_device("test-custom", mock_device)

        assert "test-custom" in _device_registry
        assert get_device("test-custom") is mock_device

        # Cleanup
        del _device_registry["test-custom"]

    def test_get_device_not_found(self):
        """Test getting a non-existent device raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            get_device("nonexistent-device")

        assert "nonexistent-device" in str(exc_info.value)
        assert "not registered" in str(exc_info.value)

    def test_get_device_local(self):
        """Test getting the built-in local device."""
        device = get_device("local")

        assert device.name == "local"
        assert device.capabilities.isolation_level == "none"
        assert device.capabilities.effect_capture == "git"

    def test_get_device_container(self):
        """Test getting the built-in container device."""
        device = get_device("container")

        assert device.name == "container"
        assert device.capabilities.isolation_level == "container"
        assert device.capabilities.effect_capture == "overlay"

    def test_list_devices(self):
        """Test listing all registered devices."""
        devices = list_devices()

        assert "local" in devices
        assert "container" in devices

    def test_scope_current_device_none_does_not_instantiate_container(self, monkeypatch):
        """Checking current_device without a Device context should not probe container infrastructure."""

        @dataclass
        class SentinelContainerDevice:
            name: str = "container"
            capabilities: DeviceCapabilities = field(
                default_factory=lambda: DeviceCapabilities(
                    isolation_level="container",
                    effect_capture="overlay",
                    supports_checkpoint=False,
                    supports_restore=False,
                    supports_dmtcp=False,
                    supports_parallel=True,
                )
            )

        original = _device_registry.pop("container", None)
        monkeypatch.setattr("shepherd_runtime.device.ContainerDevice", SentinelContainerDevice)

        try:
            with Scope() as scope:
                assert scope.current_device is None
                assert "container" in list_devices()
                assert "container" not in _device_registry
        finally:
            _device_registry.pop("container", None)
            if original is not None:
                _device_registry["container"] = original


class TestDeviceContextManager:
    """Tests for the Device() context manager."""

    def test_device_context_sets_current(self):
        """Test that Device() sets the current device."""
        assert get_current_device() is None

        with Device("local") as device:
            assert get_current_device() is device
            assert device.name == "local"

        # Should be reset after exiting
        assert get_current_device() is None

    def test_device_context_nested_raises_error(self):
        """Test that nested Device() contexts raise DeviceNestingError.

        Device nesting is prohibited to prevent confusion about which device
        is handling execution. If you need to change devices, exit the current
        device context first.
        """
        from shepherd_runtime.device.errors import DeviceNestingError

        with Device("local") as outer:
            assert get_current_device() is outer
            assert outer.name == "local"

            # Attempting to nest should raise
            with pytest.raises(DeviceNestingError) as exc_info, Device("container"):
                pass

            assert "Cannot nest" in str(exc_info.value)

            # Outer device should still be active
            assert get_current_device() is outer

        # Should be None after context exits
        assert get_current_device() is None

    def test_device_context_yields_device(self):
        """Test that Device() yields the device instance."""
        with Device("local") as device:
            assert isinstance(device, LocalDevice)
            assert device.name == "local"

    def test_device_context_raises_for_unknown(self):
        """Test that Device() raises for unknown device names."""
        with pytest.raises(KeyError), Device("unknown-device"):
            pass


class TestDeviceContextVar:
    """Tests for the device context variable behavior."""

    def test_context_var_default_none(self):
        """Test that the default context var value is None."""
        assert get_current_device() is None

    def test_context_var_thread_local(self):
        """Test that context var is thread-local."""
        import threading

        results = {}

        def thread_func(device_name):
            with Device(device_name):
                results[device_name] = get_current_device().name

        # Run in separate threads
        t1 = threading.Thread(target=thread_func, args=("local",))
        t2 = threading.Thread(target=thread_func, args=("container",))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Each thread should have seen its own device
        assert results["local"] == "local"
        assert results["container"] == "container"


class TestLocalDevice:
    """Tests for the LocalDevice implementation."""

    def test_local_device_properties(self):
        """Test LocalDevice has correct properties."""
        device = LocalDevice()

        assert device.name == "local"
        assert device.capabilities.isolation_level == "none"
        assert device.capabilities.effect_capture == "git"
        assert device.capabilities.supports_parallel is True

    @pytest.mark.asyncio
    async def test_local_device_create_sandbox(self):
        """Test LocalDevice.create_sandbox() returns a handle."""
        from shepherd_core.foundation.protocols.device import SandboxConfig

        device = LocalDevice()
        mock_scope = MagicMock()
        config = SandboxConfig(context_states={})

        sandbox = await device.create_sandbox(mock_scope, config)

        assert sandbox.sandbox_id is not None
        assert sandbox.device_name == "local"

    @pytest.mark.asyncio
    async def test_local_device_execute(self):
        """Test LocalDevice.execute() returns a result."""
        from shepherd_core.foundation.protocols.device import (
            ExecutionSpec,
        )
        from shepherd_runtime.device.local import LocalSandboxHandle

        device = LocalDevice()
        sandbox = LocalSandboxHandle(sandbox_id="test-123")
        spec = ExecutionSpec(prompt="test prompt", provider_config={})

        result = await device.execute(sandbox, spec)

        assert result.success is True
        assert result.metadata["device"] == "local"

    @pytest.mark.asyncio
    async def test_local_device_extract_effects(self):
        """Test LocalDevice.extract_effects() returns empty bundle."""
        from shepherd_core.foundation.protocols.device import ExecutionResult
        from shepherd_runtime.device.local import LocalSandboxHandle

        device = LocalDevice()
        sandbox = LocalSandboxHandle(sandbox_id="test-123")
        result = ExecutionResult(success=True)

        bundle = await device.extract_effects(sandbox, result)

        assert bundle.context_effects == {}
        assert bundle.lifecycle_effects == []

    @pytest.mark.asyncio
    async def test_local_device_cleanup(self):
        """Test LocalDevice.cleanup() is a no-op."""
        from shepherd_runtime.device.local import LocalSandboxHandle

        device = LocalDevice()
        sandbox = LocalSandboxHandle(sandbox_id="test-123")

        # Should not raise
        await device.cleanup(sandbox)


class TestDeviceContextCleanup:
    """Tests for sandbox cleanup on Device context exit.

    These tests verify that container sandboxes are cleaned up when the
    Device context manager exits, preventing overlay mount leaks.
    """

    def test_device_context_cleans_sandboxes_on_exit(self):
        """Device context should clean up sandboxes created during the context."""
        from shepherd_runtime.scope import Scope

        with Scope() as scope:
            # Access sandbox tracker's internal dict
            sandboxes = scope._sandbox_tracker._sandboxes

            # Record sandbox count before
            initial_count = len(sandboxes)

            # Simulate sandboxes being created during Device context
            mock_sandbox = MagicMock()
            mock_sandbox.cleanup = MagicMock()

            with Device("local"):
                # Simulate a sandbox being registered (as ContainerDevice does)
                sandboxes["test-sandbox-123"] = mock_sandbox

            # After Device exit, the sandbox should have been cleaned up
            mock_sandbox.cleanup.assert_called_once()
            assert "test-sandbox-123" not in sandboxes

    def test_device_context_preserve_overlays_skips_cleanup(self):
        """Device context with preserve_overlays=True should not clean sandboxes."""
        from shepherd_runtime.scope import Scope

        with Scope() as scope:
            sandboxes = scope._sandbox_tracker._sandboxes

            mock_sandbox = MagicMock()
            mock_sandbox.cleanup = MagicMock()

            with Device("local", preserve_overlays=True):
                sandboxes["test-sandbox-456"] = mock_sandbox

            # Sandbox should NOT have been cleaned up
            mock_sandbox.cleanup.assert_not_called()
            assert "test-sandbox-456" in sandboxes

            # Manual cleanup for test
            del sandboxes["test-sandbox-456"]

    def test_device_context_only_cleans_sandboxes_created_during_context(self):
        """Device context should only clean sandboxes created during that context."""
        from shepherd_runtime.scope import Scope

        with Scope() as scope:
            sandboxes = scope._sandbox_tracker._sandboxes

            # Pre-existing sandbox (created before Device context)
            pre_existing_sandbox = MagicMock()
            pre_existing_sandbox.cleanup = MagicMock()
            sandboxes["pre-existing"] = pre_existing_sandbox

            # New sandbox created during context
            new_sandbox = MagicMock()
            new_sandbox.cleanup = MagicMock()

            with Device("local"):
                sandboxes["new-sandbox"] = new_sandbox

            # Only the new sandbox should have been cleaned
            new_sandbox.cleanup.assert_called_once()
            pre_existing_sandbox.cleanup.assert_not_called()

            # Pre-existing should still be there
            assert "pre-existing" in sandboxes
            assert "new-sandbox" not in sandboxes

            # Manual cleanup
            del sandboxes["pre-existing"]

    def test_device_context_handles_cleanup_exceptions(self):
        """Device context should handle exceptions during sandbox cleanup gracefully."""
        from shepherd_runtime.scope import Scope

        with Scope() as scope:
            sandboxes = scope._sandbox_tracker._sandboxes

            failing_sandbox = MagicMock()
            failing_sandbox.cleanup = MagicMock(side_effect=RuntimeError("Cleanup failed"))

            with Device("local"):
                sandboxes["failing-sandbox"] = failing_sandbox

            # Should not raise - cleanup errors are logged but don't propagate
            # Sandbox should be removed from tracking even if cleanup fails
            assert "failing-sandbox" not in sandboxes

    def test_device_context_uses_discard_if_no_cleanup(self):
        """Device context should use discard() if cleanup() is not available."""
        from shepherd_runtime.scope import Scope

        with Scope() as scope:
            sandboxes = scope._sandbox_tracker._sandboxes

            # Sandbox with discard() but not cleanup()
            sandbox_with_discard = MagicMock(spec=["discard"])
            sandbox_with_discard.discard = MagicMock()

            with Device("local"):
                sandboxes["discard-sandbox"] = sandbox_with_discard

            # Should have called discard() instead
            sandbox_with_discard.discard.assert_called_once()
            assert "discard-sandbox" not in sandboxes


class TestDeviceContextWithOptions:
    """Tests for Device context manager with kwargs."""

    def test_device_with_debug_creates_new_instance(self):
        """Device with debug=True should create a new device instance."""
        from shepherd_runtime.device.container import ContainerDevice

        with Device("container", debug=True) as device:
            assert isinstance(device, ContainerDevice)
            assert device.debug is True

    def test_preserve_overlays_kwarg_not_passed_to_device(self):
        """preserve_overlays should be consumed by Device, not passed to device constructor."""
        # This should not raise - preserve_overlays is handled by Device()
        with Device("local", preserve_overlays=True) as device:
            assert device.name == "local"
            # LocalDevice doesn't have preserve_overlays attribute
            assert not hasattr(device, "preserve_overlays")
