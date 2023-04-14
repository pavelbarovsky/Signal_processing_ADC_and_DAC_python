import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ЗАДАНИЕ 1
Amp = 5  # амплитуда
f = [20, 40, 60, 80, 100]  # частоты в Гц
Umax = 10  # максимальное значение АЦП
Umin = -10  # минимальное значение АЦП
r = 8  # число бит в АЦП
Np = 128  # колво точек для графика
N_ACD = 64  # колво точек для графика АЦП

T = 0.001  # период дискретизации
N = 1024  # количество отсчетов
t = np.linspace(0, T * N, N)
x = np.zeros_like(t)

# ЗАДАНИЕ 2
for freq in f:
    x += Amp * np.sin(2 * np.pi * freq * t)

plt.plot(t, x, color='coral')
plt.title('Полигармонический сигнал')
plt.xlabel('Время, сек')
plt.ylabel('X(t)')
plt.grid(True)
plt.show()

# ЗАДАНИЕ 3
Vref = Umax - Umin  # опорное напряжение
A = Vref / (2 ** r - 1)  # коэффициент преобразования
B = Umin  # смещение
# дискретный сигнал, используя заданные параметры АЦП:
c_ACD = np.round((x - B) / A) * A + B

plt.figure()
plt.plot(t, c_ACD, color='indigo')
plt.title('Сигнал АЦП')
plt.xlabel('Время, сек')
plt.ylabel('C(t)')
plt.grid(True)
plt.show()

# ЗАДАНИЕ 4
error_x = x - c_ACD

plt.plot(t, error_x, color='mediumvioletred')
plt.title('Ошибка квантования e(t)')
plt.xlabel('Время, сек')
plt.ylabel('Ошибка, е(t)')
plt.grid(True)
plt.show()

print('Статистические характеристики ошибки между начальным и АЦП сигналами: ')
print(f'Среднее арифметическое = {np.mean(error_x):.3f}')
print(f'Среднеквадратичное отклонение = {np.std(error_x):.3f}')
print(f'Дисперсия = {np.var(error_x):.3f}')
print('\n')

# Задание 5
noise = np.random.normal(0, Amp * 0.2, N)  # шум
x_noise = x + noise  # сигнал с шумом
c_ACD_noise = c_ACD + noise

plt.plot(t, noise, color='hotpink')
plt.title('Шум')
plt.xlabel('Время, сек')
plt.ylabel('Амплитуда, Вт')
plt.grid(True)
plt.show()

plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.plot(t[:Np], x_noise[:Np], color='coral', label="Сигнал с шумом")
plt.plot(t[:Np], x[:Np], color='indigo', label="Исходный сигнал", alpha=0.5)
plt.title('Полигармонический сигнал с шумом')
plt.xlabel('Время, сек')
plt.ylabel('Амплитуда, Вт')
plt.legend(loc=1)
plt.grid(True)
plt.show()

plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.subplot(2, 1, 1)
plt.plot(t, x_noise, color='coral')
plt.title('Полигармонический сигнал с шумом')
plt.xlabel('Время, сек')
plt.ylabel('X(t)')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(t, c_ACD_noise, color='indigo')
plt.title('Сигнал АЦП с шумом')
plt.xlabel('Время, сек')
plt.ylabel('C(t)')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.show()

# Задание 6
c_ACD_noise_fft = np.fft.fft(c_ACD_noise)
freq = np.fft.fftfreq(N, T)

b = 1 / T
Amplitude = 2 * np.abs(c_ACD_noise_fft) / N
Amplitude = Amplitude[:int(N / 2)]
freq = freq[:int(N / 2)]
Amplitude = Amplitude[freq >= 0]
freq = freq[freq >= 0]

plt.plot(freq[:Np], Amplitude[:Np], color='coral')
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда, Вт')
plt.title('Амплитудный спектр сигнала')
plt.grid(True)
plt.show()

# Задание 7
L = 3
F = np.where(2 * np.abs(c_ACD_noise_fft) / N > L, 1, 0)

plt.plot(freq[:Np], F[:Np], color='mediumvioletred')
plt.title('Очищенный амлитудный спектр сигнала АЦП')
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда, Вт')
plt.grid(True)
plt.show()

# Задание 8 и 9
c_filt = np.fft.fft(c_ACD_noise)
c_ACD_noise_regen = np.real(np.fft.ifft(c_filt * F))

# Задание 10
plt.plot(t[:N_ACD], c_ACD[:N_ACD], color='coral', label='Исходный сигнал АЦП')
plt.plot(t[:N_ACD], c_ACD_noise[:N_ACD], color='olive', label='Сигнал АЦП с шумом', alpha=0.8)
plt.plot(t[:N_ACD], c_ACD_noise_regen[:N_ACD], color='indigo', label='Восстановленный очищенный сигнал АЦП')
plt.xlabel('Время, сек')
plt.ylabel('Амплитуда, Вт')
plt.title('Графики АЦП')
plt.legend()
plt.grid(True)
plt.show()

error_ACD_1 = c_ACD_noise - c_ACD
print('Статистические характеристики ошибки между сигналами АЦП и АЦП с шумом: ')
print(f'Среднее арифметическое = {np.mean(error_ACD_1):.3f}')
print(f'Среднеквадратичное отклонение = {np.std(error_ACD_1):.3f}')
print(f'Дисперсия = {np.var(error_ACD_1):.3f}')
print('\n')
error_ACD_2 = c_ACD_noise - c_ACD_noise_regen
print('Статистические характеристики ошибки между сигналами АЦП и АЦП восстановленного: ')
print(f'Среднее арифметическое = {np.mean(error_ACD_2 ):.3f}')
print(f'Среднеквадратичное отклонение = {np.std(error_ACD_2 ):.3f}')
print(f'Дисперсия = {np.var(error_ACD_2):.3f}')
print('\n')

print(f'Уровень фильтрации L = {L}')
print(f'MAE для зашумленного сигнала АЦП: {mean_absolute_error(c_ACD_noise_regen, c_ACD):.3f}')
print(f'MSE для зашумленного сигнала АЦП: {mean_squared_error(c_ACD_noise_regen, c_ACD):.3f}')
print(f'R2 для зашумленного сигнала АЦП: {r2_score(c_ACD_noise_regen, c_ACD):.3f}')
print(f'MAE для очищенного сигнала АЦП: {mean_absolute_error(c_ACD_noise_regen, c_ACD_noise):.3f}')
print(f'MSE для очищенного сигнала АЦП: {mean_squared_error(c_ACD_noise_regen, c_ACD_noise):.3f}')
print(f'R2 для очищенного сигнала АЦП: {r2_score(c_ACD_noise_regen, c_ACD_noise):.3f}')
print('\n')

plt.plot(t, error_ACD_1, color='darkorchid')
plt.title('График ошибки между сигналами АЦП и АЦП с шумом')
plt.xlabel('Время, сек')
plt.ylabel('Ошибка, e(t)')
plt.grid(True)
plt.show()

plt.plot(t, error_ACD_2, color='deepskyblue')
plt.title('График ошибки между сигналами АЦП и АЦП восстановленного')
plt.xlabel('Время, сек')
plt.ylabel('Ошибка, e(t)')
plt.grid(True)
plt.show()

# Задание 11
r_DAC = 8  # Количество бит ЦАП
Vref_DAC = 5  # Опорное напряжение ЦАП
max_code = 2 ** r_DAC - 1  # максимальное значение кода ЦАП
DAC_step = Vref_DAC / max_code  # шаг квантования

# Квантуем сигнал
x_DAC = np.round(c_ACD / DAC_step) * DAC_step
x_DAC_noise = np.round(c_ACD_noise / DAC_step) * DAC_step
x_DAC_noise_regen = np.round(c_ACD_noise_regen / DAC_step) * DAC_step

# Задание 12
plt.plot(t[:N_ACD], x_DAC[:N_ACD], color='blue', label='Исходный сигнал АЦП через ЦАП')
plt.plot(t[:N_ACD], x_DAC_noise[:N_ACD], color='darkorange', label='Сигнал АЦП с шумом через ЦАП', alpha=0.4)
plt.plot(t[:N_ACD], x_DAC_noise_regen[:N_ACD], '--k', label='Восстановленный очищенный сигнал АЦП через ЦАП')
plt.xlabel('Время, сек')
plt.ylabel('Амплитуда, Вт')
plt.title('Графики ЦАП')
plt.legend()
plt.grid(True)
plt.show()

# Визуализируем результат
plt.plot(t[:Np], x[:Np], color='crimson', label='Начальный полигарм.сигнал')
plt.plot(t[:Np], x_DAC_noise_regen[:Np], '--k', label='Восстановленный очищенный сигнал АЦП через ЦАП', alpha=0.5)
plt.xlabel('Время, сек')
plt.ylabel('Амплитуда, Вт')
plt.grid(True)
plt.legend()
plt.show()

# Задание 13
error_DAC_1 = x_DAC_noise - x_DAC
print('Статистические характеристики ошибки между сигналами ЦАП и ЦАП с шумом: ')
print(f'Среднее арифметическое = {np.mean(error_DAC_1):.3f}')
print(f'Среднеквадратичное отклонение = {np.std(error_DAC_1):.3f}')
print(f'Дисперсия = {np.var(error_DAC_1):.3f}')
print('\n')

error_DAC_2 = x_DAC_noise_regen - x_DAC
print('Статистические характеристики ошибки между сигналами ЦАП и ЦАП восстановленного: ')
print(f'Среднее арифметическое = {np.mean(error_DAC_2):.3f}')
print(f'Среднеквадратичное отклонение = {np.std(error_DAC_2):.3f}')
print(f'Дисперсия = {np.var(error_DAC_2):.3f}')
print('\n')

plt.plot(t, error_DAC_1, color='teal')
plt.title('График ошибки между сигналами ЦАП и ЦАП с шумом')
plt.xlabel('Время, сек')
plt.ylabel('Ошибка, e(t)')
plt.grid(True)
plt.show()

plt.plot(t, error_DAC_2, color='tomato')
plt.title('График ошибки между сигналами ЦАП и ЦАП восстановленного')
plt.xlabel('Время, сек')
plt.ylabel('Ошибка, e(t)')
plt.grid(True)
plt.show()
