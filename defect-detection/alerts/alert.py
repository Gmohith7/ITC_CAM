import time
import threading
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _init_gpio():
    """
    Return (led, buzzer) using gpiozero with the lgpio pin factory.
    Pi 5 uses the RP1 I/O chip (chip index 4); gpiozero's default pigpio/RPiGPIO
    backends do NOT work on Pi 5. lgpio does.
    """
    from gpiozero.pins.lgpio import LGPIOFactory
    from gpiozero import Device, LED, Buzzer
    Device.pin_factory = LGPIOFactory(chip=4)   # RP1 chip on Pi 5
    led = LED(config.GPIO_LED_PIN)
    buzzer = Buzzer(config.GPIO_BUZZER_PIN)
    return led, buzzer


class AlertSystem:
    """
    GPIO LED + buzzer alerts. Silently no-ops when GPIO is unavailable (dev machine).

    trigger() is non-blocking: it fires the alert in a background thread so the
    OCR worker is never stalled. A debounce lock prevents overlapping triggers.
    """

    def __init__(self):
        self._available = False
        self._lock = threading.Lock()
        self._active = False
        try:
            self.led, self.buzzer = _init_gpio()
            self._available = True
            print("[Alert] GPIO initialised (lgpio / RP1).")
        except Exception as e:
            print(f"[Alert] GPIO unavailable ({e}). Running in no-op mode.")

    def trigger(self, duration: float = None):
        """
        Fire LED + buzzer for `duration` seconds (default: config.ALERT_DURATION_S).
        Non-blocking — runs in a background thread. Debounced: if an alert is
        already active the call is a no-op.
        """
        duration = duration if duration is not None else config.ALERT_DURATION_S
        if not self._lock.acquire(blocking=False):
            return  # already alerting
        self._active = True
        threading.Thread(target=self._alert_thread, args=(duration,), daemon=True).start()

    def _alert_thread(self, duration: float):
        try:
            if self._available:
                self.led.on()
                self.buzzer.on()
                time.sleep(duration)
                self.led.off()
                self.buzzer.off()
            else:
                print(f"[Alert] DEFECT — would trigger GPIO for {duration:.1f}s.")
                time.sleep(duration)
        finally:
            self._active = False
            self._lock.release()

    def clear(self):
        """Immediately silence outputs (safe to call while alert is active)."""
        if self._available:
            self.led.off()
            self.buzzer.off()
