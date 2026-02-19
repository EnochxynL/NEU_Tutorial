# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from Box2D.examples.framework import (Framework, Keys, main)
from Box2D import (b2EdgeShape, b2FixtureDef, b2PolygonShape, b2CircleShape)

class Control():
    """ This should be an abstract class. Do not instantiate."""
    DELTA_T = 0.01

    def get_xa(self, xe):
        raise NotImplementedError()

class PControl(Control):
    """ I control unit"""
    def __init__(self, kp=1):
        """set Kp
        Parameters:
        kp(float): Kp"""
        self._kp = kp

    def reset(self, kp):
        """ resets the unit """
        self._kp = kp

    def get_xa(self, xe):
        """ give input, get output
        Parameters:
        xe(float): input xe

        Returns:
        float: output xa """
        return xe * self._kp

class IControl(Control):
    """ I control unit"""
    def __init__(self, ki=1):
        """ set Ki
        Parameters:
        ki(float): Ki """
        self._ki = ki
        self._sum = 0

    def reset(self, ki):
        """ resets the unit """
        self._ki = ki
        self._sum = 0

    def get_xa(self, xe):
        """ give input, get output
        Parameters:
        xe(float): input xe

        Returns:
        float: output xa """
        self._sum = self._sum + xe
        return self._ki * self._sum * self.DELTA_T

class DControl(Control):
    """ D control unit """
    def __init__(self, kd=1):
        """ set Kd
        Parameters:
        kd(float): Kd """
        self._kd = kd
        self._xe_old = 0

    def reset(self, kd):
        """ resets the unit """
        self._kd = kd
        self._xe_old = 0

    def get_xa(self, xe):
        """ give input, get output
        Parameters:
        xe(float): input xe

        Returns:
        float: output xa """
        xa = self._kd * ((xe - self._xe_old) / self.DELTA_T)
        self._xe_old = xe
        return xa

class PIDControl():
    """ PID controller """
    def __init__(self, kp, ki, kd):
        """ set Kp, Ki, Kd
        Parameters:
        kp(float): Kp
        ki(float): Ki
        kd(float): Kd """
        self._pControl = PControl(kp)
        self._iControl = IControl(ki)
        self._dControl = DControl(kd)

    def update_params(self, kp, ki, kd):
        """ update Kp, Ki, Kd
        Parameters:
        kp(float): Kp
        ki(float): Ki
        kd(float): Kd """
        self._pControl.reset(kp)
        self._iControl.reset(ki)
        self._dControl.reset(kd)

    def get_xa(self, xe):
        """ give input, get output
        Parameters:
        xe(float): input xe

        Returns:
        float: output xa """
        xa = self._pControl.get_xa(xe) + self._iControl.get_xa(xe) + self._dControl.get_xa(xe)
        return xa


class BodyPendulum(Framework):
    name = "Inverted Pendulum (PyConSys)"
    description = "(m) manual, (a) automatic, (n) new world"
    speed = 3

    def __init__(self):
        super(BodyPendulum, self).__init__()

        self.createWorld()

    def createWorld(self):
        self._isLiving = True
        self._auto = False
        self._pid_control = PIDControl(105, 83, 28)  # kp, ki, kd


        self.ground = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-25, 0), (25, 0)])
        )

        self.carBody = self.world.CreateDynamicBody(
            position=(0, 3),
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(box=(5, 1)), density=1)

        )

        self.carLwheel = self.world.CreateDynamicBody(
            position=(-3, 1),
            fixtures=b2FixtureDef(
                shape=b2CircleShape(radius=1), density=2, friction=1)

        )

        self.carRwheel = self.world.CreateDynamicBody(
            position=(3, 1),
            fixtures=b2FixtureDef(
                shape=b2CircleShape(radius=1), density=2, friction=1)

        )

        self.pendulum = self.world.CreateDynamicBody(
            position=(0, 13),
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(box=(0.5, 10)), density=1),

        )

        self.pendelumJoin = self.world.CreateRevoluteJoint(
            bodyA=self.carBody,
            bodyB=self.pendulum,
            anchor=(0, 3),
            maxMotorTorque=1,
            enableMotor=True
        )

        self.pendelumRJoin =self.world.CreateRevoluteJoint(
            bodyA=self.carBody,
            bodyB=self.carRwheel,
            anchor=(3, 1),
            maxMotorTorque=1,
            enableMotor=True,
            #motorSpeed=10
        )

        self.pendelumLJoin = self.world.CreateRevoluteJoint(
            bodyA=self.carBody,
            bodyB=self.carLwheel,
            anchor=(-3, 1),
            maxMotorTorque=1,
            enableMotor=True,
            #motorSpeed=10
        )


    def destroyWorld(self):
        self.world.DestroyBody(self.carBody)
        self.world.DestroyBody(self.carLwheel)
        self.world.DestroyBody(self.carRwheel)
        self.world.DestroyBody(self.pendulum)
        self._isLiving = False


    def Keyboard(self, key):
        if key == Keys.K_a:
            if self._isLiving:
                self._auto = True

        elif key == Keys.K_m:
            self._auto = False
            if self._isLiving:
                self.pendelumLJoin.motorSpeed = 0
                self.pendelumLJoin.maxMotorTorque = 1
                self.pendelumRJoin.motorSpeed = 0
                self.pendelumRJoin.maxMotorTorque = 1
        elif key == Keys.K_n:
            if self._isLiving:
                self.destroyWorld()
                self.createWorld()


    def Step(self, settings):
        super(BodyPendulum, self).Step(settings)

        w = 0
        e = (w - self.pendulum.angle*-1)
        y = self._pid_control.get_xa(e)

        if self._auto and self._isLiving:
            self.pendelumLJoin.maxMotorTorque = 1000
            self.pendelumRJoin.maxMotorTorque = 1000
            self.pendelumLJoin.motorSpeed = y
            self.pendelumRJoin.motorSpeed = y


if __name__ == "__main__":
    main(BodyPendulum)