# Implementation Notes
# # Controller Class 
class JoystickControl(object):
    def __init__(self, axis_range=2, axis_scale=3.0):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.DEADBAND, self.AXIS_RANGE, self.AXIS_SCALE = 0.1, axis_range, axis_scale

    def input(self):
        pygame.event.get()
        zs = []

        # Latent Actions / 2D End-Effector Control
        if self.AXIS_RANGE == 2:
            for i in range(3, 3 + self.AXIS_RANGE):
                z = self.gamepad.get_axis(i)
                if abs(z) < self.DEADBAND:
                    z = 0.0
                zs.append(z * self.AXIS_SCALE)

                
# Secret, Tri-Axial End Effector Control
        else:
            for i in range(self.AXIS_RANGE):
                z = self.gamepad.get_axis(i)
                if abs(z) < self.DEADBAND:
                    z = 0.0
                zs.append(z * self.AXIS_SCALE)

        # Button Press
        a, b = self.gamepad.get_button(0), self.gamepad.get_button(1)
        x, y, stop = self.gamepad.get_button(2), self.gamepad.get_button(3), self.gamepad.get_button(7)
        
        return zs, a, b, x, y, stop
