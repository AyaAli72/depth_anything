import numpy as np
import math

def collision(Oobs, Probi):
    for obs in Oobs:
        obs_dist = math.sqrt((Probi[0] - obs[0])**2 + (Probi[1] - obs[1])**2)
        if obs_dist < 0.5:
            return True
    return False    

def calc_vm(obstacles):
   dmax = 5.0
   vmf = [0 if obs_dist <= dmax else 1 for obs_dist in np.linalg.norm(obstacles, axis= 1) ]
   return min(vmf)

def check_velocity(vdwv, omegamax_dwvf):
   vmax = 1.0
   omegamax = math.radians(30)
   return min(vdwv, vmax), min(omegamax_dwvf, omegamax)

def one_path(vdwv, omega_dwvf, theta_prev, x_prev, y_prev, dt ):
   theta_next = theta_prev + omega_dwvf * dt
   x_next = x_prev + vdwv * math.cos(theta_next) * dt
   y_next = y_prev + vdwv * math.sin(theta_next) * dt
   return x_next, y_next, theta_next

def dwa_path_planning(vini, omega_ini, delta_v, delta_omega, Ntlv, Nagv, fmax, obstacles, dt):
    g, i = 1, 1
    Orob = None

    while Ntlv > g:
        vdwa = vini + delta_v * (g - 1)

        h = 1
        while Nagv > h:
            omega_dwa = omega_ini + delta_omega * (h - 1)

            f = 1
            while fmax > f:
                # Obstacle positions
                Oobs = obstacles

                # Calculate virtual modification factor
                omega_vmf = calc_vm(Oobs)

                # Velocity modification
                vdwv = vdwa
                if f == 1:
                    omega_dwvf = omega_dwa + omega_vmf
                else:
                    omega_dwvf = omega_dwvf - 1 + omega_vmf

                # Check velocities
                vdwv, omega_dwvf = check_velocity(vdwv, omega_dwvf)

                # Simulate one path segment
                xrobf, yrobf, theta_robf = one_path(vdwv, omega_dwvf, 0, 0, 0, dt)  # Update states

                Probi = (xrobf, yrobf, theta_robf)  # Current candidate path

                # Collision detection
                if collision(Oobs, Probi):
                    Probi = None
                else:
                    Orob = Probi
                    i += 1

                f += 1

            h += 1

        g += 1

    return Orob

      


