import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _init_gpio():
    """
    Return (led, buzzer) using gpiozero with the lgpio pin factory.
    Pi 5 uses the RP1 I/O chip (chip index 4) — gpiozero's default pigpio/RPiGPIO
    backends do NOT work on Pi 5. lgpio does.
    RPi.GPIO is not used at all: it is incompatible with Pi 5.
    """
    from gpiozero.pins.lgpio import LGPIOFactory
    from gpiozero import Device, LED, Buzzer
    Device.pin_factory = LGPIOFactory(chip=4)   # RP1 chip on Pi 5
    led = LED(config.GPIO_LED_PIN)
    buzzer = Buzzer(config.GPIO_BUZZER_PIN)
    return led, buzzer


class AlertSystem:
    """GPIO LED + buzzer alerts. Silently no-ops when GPIO is unavailable (dev machine)."""

    def __init__(self):
        self._available = False
        try:
            self.led, self.buzzer = _init_gpio()
            self._available = True
            print("[Alert] GPIO initialised (lgpio / RP1).")
        except Exception as e:
            print(f"[Alert] GPIO unavailable ({e}). Running in no-op mode.")

    def trigger(self, duration: float = 1.0):
        if self._available:
            self.led.on()
            self.buzzer.on()
            time.sleep(duration)
            self.led.off()
            self.buzzer.off()
        else:
            print(f"[Alert] DEFECT — would trigger GPIO for {duration}s.")

    def clear(self):
        if self._available:
            self.led.off()
            self.buzzer.off()
