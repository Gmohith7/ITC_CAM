import time
import threading
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _init_gpio():
    """
    Return (led, buzzer) using gpiozero + lgpio backend.
    Pi 5 uses the RP1 I/O chip (chip index 4).
    """
    from gpiozero.pins.lgpio import LGPIOFactory
    from gpiozero import Device, LED, Buzzer
    Device.pin_factory = LGPIOFactory(chip=4)
    led    = LED(config.GPIO_LED_PIN)
    buzzer = Buzzer(config.GPIO_BUZZER_PIN)
    return led, buzzer


class AlertSystem:
    """
    GPIO LED + buzzer alerts. Silently no-ops when GPIO is unavailable.

    trigger() is non-blocking (fires in a daemon thread).
    A lock prevents overlapping triggers (debounce).
    clear() is safe to call at any time — it never races with the alert thread.
    """

    def __init__(self):
        self._available = False
        self._lock      = threading.Lock()
        self._active    = False
        try:
            self.led, self.buzzer = _init_gpio()
            self._available = True
            print("[Alert] GPIO initialised (lgpio / RP1).")
        except Exception as e:
            print(f"[Alert] GPIO unavailable ({e}). Running in no-op mode.")

    def trigger(self, duration: float = None):
        """
        Fire LED + buzzer for `duration` seconds (default: config.ALERT_DURATION_S).
        Non-blocking. No-op if an alert is already running.
        """
        duration = duration if duration is not None else config.ALERT_DURATION_S
        if not self._lock.acquire(blocking=False):
            return   # already alerting — debounce
        self._active = True
        threading.Thread(target=self._alert_thread, args=(duration,), daemon=True).start()

    def _alert_thread(self, duration: float):
        try:
            if self._available:
                try:
                    self.led.on()
                    self.buzzer.on()
                except Exception as e:
                    print(f"[Alert] GPIO on() failed: {e}")
                time.sleep(duration)
                try:
                    self.led.off()
                    self.buzzer.off()
                except Exception as e:
                    print(f"[Alert] GPIO off() failed: {e}")
            else:
                if duration > 0:
                    print(f"[Alert] DEFECT — would trigger GPIO for {duration:.1f}s.")
                time.sleep(max(0, duration))
        finally:
            self._active = False
            self._lock.release()

    def clear(self):
        """
        Silence outputs immediately. Safe to call at any time.
        If an alert thread is active it will still finish its sleep and
        then turn off cleanly — we just also attempt to turn off here.
        """
        if self._available:
            try:
                self.led.off()
                self.buzzer.off()
            except Exception:
                pass  # device may already be off or closed; ignore
