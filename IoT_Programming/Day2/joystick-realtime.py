from sense_hat import SenseHat
from datetime import datetime

hat = SenseHat()

led_state = {}
while True:
    for event in hat.stick.get_events():
        if event.direction == "up" and \
           (event.action == "pressed" or event.action == "held"):
            if not "letter" in led_state :
                hat.show_letter("u")
                led_state["letter"] = "u"
            led_state["timestamp"] = datetime.now().timestamp()
        else:
            print(event)
    if "letter" in led_state and \
       datetime.now().timestamp() - led_state["timestamp"] > 1.0 :
        hat.show_letter(" ")
        led_state.clear()

hat.clear()
