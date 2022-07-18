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

# TODO
        # Latent Actions / 2D End-Effector Control
        # when in resting state

        # right stick?

        # x = zs[0]
        # left - 
        # right +

        # y = zs[1]
        # forward +
        # backward -

        if self.AXIS_RANGE == 2:
            # range 3-5
            for i in range(3, 3 + self.AXIS_RANGE):
                # print("i: ", i)
                z = self.gamepad.get_axis(i)
                if abs(z) < self.DEADBAND:
                    # centered axis
                    z = 0.0
                # print("z right: ", z)
                zs.append(z * self.AXIS_SCALE)
                #print("zs right: ", zs)
                #print("zs right: {.2f}, {.2f}".format(zs[0]))
                print("zs: ", zs)
        
        
# TODO                
# Secret, Tri-Axial End Effector Control
        # left stick?
        # zs[4]
        # up + 
        # down - 


        else:
            ## print("Range 2: ", range(self.AXIS_RANGE))
            for i in range(self.AXIS_RANGE):
                z = self.gamepad.get_axis(i)
                if abs(z) < self.DEADBAND:
                    z = 0.0
                print("z left: ", z)
                # set the current position of the left axis?
                zs.append(z * self.AXIS_SCALE)
                #print("zs left: ", zs)

        # Button Press
        # gets the current state of buttons a, b, x, y, stop
        a, b = self.gamepad.get_button(0), self.gamepad.get_button(1)
        x, y, stop = self.gamepad.get_button(2), self.gamepad.get_button(3), self.gamepad.get_button(7)
        
        return zs, a, b, x, y, stop

if __name__ == "__main__":
    print("Running!")
    controller = Joystick()
    print("Controller: " , controller.input())
    print("zs0: ", controller.input()[0])
    print("zs1: ", controller.input()[1])
    print("zs2: ", controller.input()[2])
    print("zs3: ", controller.input()[3])
    print("zs4: ", controller.input()[4])


    # handling the inputs
    # for all buttons, check the state of the button
    while True:
        if (controller.input()[1]) == 1:
            print("A is pressed")
        if (controller.input()[2]) == 1:
            print("B is pressed")
        if (controller.input()[3]) == 1:
            print("x is pressed")
        if (controller.input()[4]) == 1:
            print("y is pressed")
        #else:
        #    print("No button pressed")


        # zs axes: 
        # negative zs values = left
        # positive zs values = right
        # right joystick

        #print("zs: ", controller.input())
        # left/right
        print(controller.input()[0][0]) 
        if (controller.input()[0][0]) > 0:
            print("right joystick: moving right")
        elif (controller.input()[0][0]) < 0:
            print("right joystick: moving left")

        # forward/backward
        print(controller.input()[0][1])
        if (controller.input()[0][1]) > 0:
            print("right joystick: moving forward")
        elif (controller.input()[0][1]) < 0:
            print("right joystick: moving backward")

        # left joystick
        # up/down

        print(controller.input()[1]) 
        if(controller.input()[1]) > 0:
            print("left joystick: moving right")
        elif (controller.input()[1]) < 0:
            print("right joystick: moving left")



        # if user presses stop button, end the program
        if (controller.input()[5]) == 1:
            print("User pressed stop")
            pygame.quit()
            exit()

        # TODO: comment
        time.sleep(1/20)


# TODO: axis zs
    #while True:
     #   if ():
