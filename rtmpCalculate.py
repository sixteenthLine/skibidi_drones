import cv2
import numpy as np
import scipy.integrate as ode

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
    rtmp_url = 'rtmp://192.168.31.23:1935/hls/live'  # Replace with your RTMP server URL

    # Open a video capture object (for example, the default camera)
    cap = cv2.VideoCapture(rtmp_url,cv2.CAP_IMAGES)  # Use 0 for the default camera, or provide a video file path

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

    # # Get the width and height of the video frames
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # # Define the codec and create a VideoWriter object for RTMP streaming
    # fourcc = cv2.VideoWriter_fourcc(*'flv1')  # 'flv1' codec for RTMP streaming
    # out = cv2.VideoWriter(rtmp_url, fourcc, 25, (frame_width, frame_height))

    fallOffset = None
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to grab frame.")
            break

        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2

        droneHeight = 100
        windSpeed = 5

        if fallOffset is None:
            fallOffset = calculateFall(droneHeight, windSpeed)
        lineLength = worldToPixel(960, droneHeight, fallOffset[0])
        radius = int(abs(worldToPixel(960, droneHeight, fallOffset[1]) - lineLength))
        center = (int(width / 2 + lineLength), height // 2)

        cv2.line(frame, (width // 2, height // 2), center, (255, 0, 0), 1)
        cv2.circle(frame, center, radius, (0, 255, 0), 2)

        cv2.imshow('Video Stream with Rectangle', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()