# Controller Class 
from pickle import TRUE
import pygame
import time

# DEADBAND less than set amount = rounded to 0

class Joystick(object):
    # default
    def __init__(self, axis_range=2, axis_scale=3.0):
        # initialize pygame module
        pygame.init()
        # creating new joystick object called 'gamepad'
        self.gamepad = pygame.joystick.Joystick(0)
        # initialize joystick module
        self.gamepad.init()
        self.DEADBAND, self.AXIS_RANGE, self.AXIS_SCALE = 0.1, axis_range, axis_scale

    def input(self):
        # print("Input")
        pygame.event.get()
        zs = []

# TODO: ignore for now
        # Latent Actions / 2D End-Effector Control
        if self.AXIS_RANGE == 2:
            for i in range(3, 3 + self.AXIS_RANGE):
                z = self.gamepad.get_axis(i)
                if abs(z) < self.DEADBAND:
                    # centered axis
                    z = 0.0
                zs.append(z * self.AXIS_SCALE)

# TODO: ignore for now                
# Secret, Tri-Axial End Effector Control
        else:
            for i in range(self.AXIS_RANGE):
                z = self.gamepad.get_axis(i)
                if abs(z) < self.DEADBAND:
                    z = 0.0
                zs.append(z * self.AXIS_SCALE)

        # Button Press
        # gets the current state of buttons a, b, x, y, stop
        a, b = self.gamepad.get_button(0), self.gamepad.get_button(1)
        x, y, stop = self.gamepad.get_button(2), self.gamepad.get_button(3), self.gamepad.get_button(7)

        # Testing with Prints
        #if a == True:
        #    print("Button 0 value is true")
        #elif a == False:
        #    print("Button 0 value is false")
        
        #print(b)
        #print(x)
        #print(y)
        
        return a, b, x, y, stop

if __name__ == "__main__":
    print("Running!")
    controller = Joystick()
    print("Controller: " , controller.input())
    
# handling the inputs
    # for all buttons, check the state of the button
    while True:
        if (controller.input()[0]) == 1:
            print("A is pressed")
        if (controller.input()[1]) == 1:
            print("B is pressed")
        if (controller.input()[2]) == 1:
            print("x is pressed")
        if (controller.input()[3]) == 1:
            print("y is pressed")
        else:
            print("No button pressed")

        # if user presses stop button, end the program
        if (controller.input()[4]) == 1:
            print("User pressed stop")
            # FIXME: pygame.error video system not initialized
            pygame.quit()

        # TODO: comment
        time.sleep(1/20)


# TODO: axis zs