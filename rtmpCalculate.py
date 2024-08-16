import os
import cv2
import numpy as np
import scipy.integrate as ode
import time

def worldToPixel(screen_w, height, length):
    w = length * 100
    dist = height * 100

    f_cm = 0.23814 # фокусная дистанция (константа)
    s_cm = 0.315 # размер матрицы (0.5 ширины) на камере (константа)
    s_w = screen_w # максимальное расширение фото/видео
    f_w = s_w * f_cm / s_cm # перевод фокусного расстояния в пиксели

    rel = w / dist # тангенс угла сдвиг броска/высоту
    res = rel * f_w # проекция на экран

    return res

def calculateFall(height, wind_speed):
    g = 9.81  # Ускорение свободного падения, м/с^2
    rho = 1.225  # Плотность воздуха, кг/м^3
    Cd = 0.47  # Коэффициент аэродинамического сопротивления (сферический объект)
    A = 0.1  # Площадь поперечного сечения, м^2
    m = 1.0  # Масса объекта, кг
    v0 = 0.0  # Начальная скорость, м/с
    wind_angle = 0  # Угол ветра, рад

    def equations(t, Y):
        x, vx, y, vy = Y
        v = np.sqrt(vx**2 + vy**2)
        Fd_x = 0.5 * Cd * rho * A * v * (vx - wind_speed * np.cos(wind_angle))
        Fd_y = 0.5 * Cd * rho * A * v * (vy - wind_speed * np.sin(wind_angle))
        ax = -Fd_x / m
        ay = -g - (Fd_y / m)
        return [vx, ax, vy, ay]

    Y0 = [0, v0, height, 0]

    sol = ode.solve_ivp(equations, [0, height], Y0, first_step = 0.01, max_step=0.01)

    x = sol.y[0]
    y = sol.y[2]

    return [x[np.where((y > -0.1) & (y < 0.1))[0][0]], x[np.where((y > 4.9) & (y < 5.1))[0][0]]]


def main():
    rtmp_url = 'rtmp://192.168.31.23:1933'
    cap = cv2.VideoCapture(rtmp_url)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    fallOffset = None
    lastTimeUpdate = time.time()
    angleDeg = 0
    while True:
        dt = time.time() - lastTimeUpdate
        ret, frame = cap.read()
        lastTimeUpdate = time.time()
        angleDeg = -45

        if not ret:
            print("Error: Failed to grab frame.")
            break

        # frame = cv2.resize(frame, (1280, 720))
        height, width = frame.shape[:2]

        droneHeight = 100
        windSpeed = 5

        if fallOffset is None:
            name = "./frame1.jpg"
            cv2.imwrite(name, frame)
            fallOffset = calculateFall(droneHeight, windSpeed)
        
        lineLength = worldToPixel(960, droneHeight, fallOffset[0])
        radius = int(abs(worldToPixel(960, droneHeight, fallOffset[1]) - lineLength))
        lineX = lineLength * np.cos(np.deg2rad(angleDeg))
        lineY = lineLength * np.sin(np.deg2rad(angleDeg))
        center = (int(width / 2 + lineX), int(height / 2 + lineY))

        cv2.line(frame, (width // 2, height // 2), center, (255, 0, 0), 1)
        cv2.circle(frame, center, radius, (0, 255, 0),  1, lineType=cv2.LINE_AA)

        cv2.imshow('Video Stream with Rectangle', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()