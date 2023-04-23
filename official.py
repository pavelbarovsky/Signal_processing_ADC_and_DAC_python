import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ЗАДАНИЕ 1
Amp = 5  # амплитуда
f = [20, 40, 60, 80, 100]  # частоты в Гц
Umax = 5  # максимальное значение АЦП
Umin = -5  # минимальное значение АЦП
r = 8  # число бит в АЦП
Np = 128  # колво точек для графика
N_ACD = 64  # колво точек для графика АЦП

T = 0.0005
N = 1024  # количество отсчетов
t = np.linspace(0, T * N, N)
x = np.zeros_like(t)

# ЗАДАНИЕ 2
for freq in f:
    x += np.sin(2 * np.pi * freq * t)

x = Amp * x / len(f)

plt.figure()
plt.plot(t, x, color='coral')
plt.title('Полигармонический сигнал')
plt.xlabel('Время, сек')
plt.ylabel('X(t)')
plt.grid(True)

# ЗАДАНИЕ 3
U_max = np.power(2, r) - 1
U_min = Umin
X_max = Amp
X_min = -Amp
c_ACD = (U_min + (x - X_min)*(U_max - U_min) / (X_max - X_min))

plt.figure()
plt.plot(t, c_ACD.round(), color='indigo')
plt.title('Сигнал АЦП')
plt.xlabel('Время, сек')
plt.ylabel('C(t)')
plt.grid(True)

# ЗАДАНИЕ 4
error_c = c_ACD.round() - c_ACD

plt.figure()
plt.plot(t, error_c, color='mediumvioletred')
plt.title('Ошибка квантования e(t)')
plt.xlabel('Время, сек')
plt.ylabel('Ошибка, е(t)')
plt.grid(True)

print('Статистические характеристики сигналом ошибки: ')
print(f'Среднее арифметическое = {np.mean(error_c):.3f}')
print(f'Среднеквадратичное отклонение = {np.std(error_c):.3f}')
print(f'Дисперсия = {np.var(error_c):.3f}')
print('\n')

# Задание 5
noise = np.random.normal(0, Amp * 0.2, N)  # шум
x_noise = x + noise  # сигнал с шумом
c_ACD_noise = c_ACD + noise

plt.figure()
plt.plot(t, noise, color='hotpink')
plt.title('Шум')
plt.xlabel('Время, сек')
plt.ylabel('Амплитуда, Вт')
plt.grid(True)

plt.figure()
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.plot(t[:Np], x_noise[:Np], color='coral', label="Сигнал с шумом")
plt.plot(t[:Np], x[:Np], color='indigo', label="Исходный сигнал", alpha=0.5)
plt.title('Полигармонический сигнал с шумом')
plt.xlabel('Время, сек')
plt.ylabel('Амплитуда, Вт')
plt.legend(loc=1)
plt.grid(True)

plt.figure()
plt.title('Сигнал АЦП с шумом')
plt.plot(t, c_ACD_noise, color='indigo', label="Выходной сигнал АЦП с шумом")
plt.ylabel('C(t)')
plt.xlabel('Время, сек')
plt.grid(True)



# Задание 6
c_ACD_noise_fft = np.fft.fft(c_ACD_noise)
freq = np.fft.fftfreq(N, T)

Amplitude = 2 * np.abs(c_ACD_noise_fft) / N
Amplitude = Amplitude[:int(N / 2)]
freq = freq[:int(N / 2)]
Amplitude = Amplitude[freq >= 0]
freq = freq[freq >= 0]

plt.figure()
plt.plot(freq[:64], Amplitude[:64], color='coral')
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда, Вт')
plt.title('Амплитудный спектр сигнала')
plt.grid(True)

# Задание 7
L = 3
F = np.where(2 * np.abs(c_ACD_noise_fft) / N > L, 1, 0)

plt.figure()
plt.plot(freq[:64], F[:64], color='mediumvioletred')
plt.title('Окно фильтра амплитудного спектра')
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда, Вт')
plt.grid(True)

# Задание 8 и 9
c_filt = np.fft.fft(c_ACD_noise)
c_ACD_noise_regen = np.real(np.fft.ifft(c_filt * F))

# Задание 10
plt.figure()
plt.plot(t[:N_ACD], c_ACD[:N_ACD], color='indigo', label='Исходный сигнал через АЦП', alpha=0.5)
plt.plot(t[:N_ACD], c_ACD_noise[:N_ACD], '--k', label='Выходной сигнал АЦП с шумом')
plt.plot(t[:N_ACD], c_ACD_noise_regen[:N_ACD], color='cornflowerblue', label='Очищенный выходной сигнал АЦП')
plt.xlabel('Время, сек')
plt.ylabel('Амплитуда, Вт')
plt.title('Графики АЦП')
plt.legend()
plt.grid(True)

error_ACD_1 = c_ACD - c_ACD_noise
print('Статистические характеристики ошибки между сигналами АЦП и АЦП с шумом: ')
print(f'Среднее арифметическое = {np.mean(error_ACD_1):.3f}')
print(f'Среднеквадратичное отклонение = {np.std(error_ACD_1):.3f}')
print(f'Дисперсия = {np.var(error_ACD_1):.3f}')
print('\n')
error_ACD_2 = c_ACD - c_ACD_noise_regen
print('Статистические характеристики ошибки между сигналами АЦП и АЦП восстановленного: ')
print(f'Среднее арифметическое = {np.mean(error_ACD_2 ):.3f}')
print(f'Среднеквадратичное отклонение = {np.std(error_ACD_2 ):.3f}')
print(f'Дисперсия = {np.var(error_ACD_2):.3f}')
print('\n')

print(f'Уровень фильтрации L = {L}')
print(f'MAE между сигналами АЦП и АЦП восстановленного: {mean_absolute_error(c_ACD, c_ACD_noise_regen):.3f}')
print(f'MSE между сигналами АЦП и АЦП восстановленного: {mean_squared_error(c_ACD, c_ACD_noise_regen):.3f}')
print(f'R2 между сигналами АЦП и АЦП восстановленного: {r2_score(c_ACD, c_ACD_noise_regen):.3f}')
print(f'MAE между сигналами АЦП и АЦП с шумом: {mean_absolute_error(c_ACD, c_ACD_noise_regen):.3f}')
print(f'MSE между сигналами АЦП и АЦП с шумом: {mean_squared_error(c_ACD, c_ACD_noise_regen):.3f}')
print(f'R2 между сигналами АЦП и АЦП с шумом: {r2_score(c_ACD, c_ACD_noise_regen):.3f}')
print('\n')

plt.figure()
plt.plot(t, error_ACD_1, color='darkorchid')
plt.title('График ошибки между сигналами АЦП и АЦП с шумом')
plt.xlabel('Время, сек')
plt.ylabel('Ошибка, e(t)')
plt.grid(True)

plt.figure()
plt.plot(t, error_ACD_2, color='deepskyblue')
plt.title('График ошибки между сигналами АЦП и АЦП восстановленного')
plt.xlabel('Время, сек')
plt.ylabel('Ошибка, e(t)')
plt.grid(True)

# Задание 11
x_DAC_noise_regen = U_min + (c_ACD_noise_regen - X_min) / (U_max - U_min) * (X_max - X_min)

# Задание 12
plt.figure()
plt.plot(t[:N_ACD], x[:N_ACD], color='blue', label='Исходный полигармонич.сигнал')
plt.plot(t[:N_ACD], x_noise[:N_ACD], color='darkorange', label='Сигнал полигармонич. с шумом', alpha=0.4)
plt.plot(t[:N_ACD], x_DAC_noise_regen[:N_ACD], 'deeppink', label='Восстановленный очищенный сигнал АЦП')
plt.xlabel('Время, сек')
plt.ylabel('Амплитуда, Вт')
plt.title('Графики ЦАП')
plt.legend()
plt.grid(True)

# Задание 13
error_DAC_1 = x - x_DAC_noise_regen
print('Статистические характеристики ошибки между сигналами исходным и восстановленным: ')
print(f'Среднее арифметическое = {np.mean(error_DAC_1):.3f}')
print(f'Среднеквадратичное отклонение = {np.std(error_DAC_1):.3f}')
print(f'Дисперсия = {np.var(error_DAC_1):.3f}')
print('\n')

error_DAC_2 = x_noise - x
print('Статистические характеристики сигнала ошибки между исходнымм сигналами с и без шума: ')
print(f'Среднее арифметическое = {np.mean(error_DAC_2):.3f}')
print(f'Среднеквадратичное отклонение = {np.std(error_DAC_2):.3f}')
print(f'Дисперсия = {np.var(error_DAC_2):.3f}')
print('\n')

print(f'MAE между исходными сигналами с и без шума: {mean_absolute_error(x, x_noise):.3f}')
print(f'MSE между исходными сигналами с и без шума: {mean_squared_error(x, x_noise):.3f}')
print(f'R2 между исходными сигналами с и без шума: {r2_score(x, x_noise):.3f}')
print(f'MAE между исходным сигналом и восстановленным: {mean_absolute_error(x, x_DAC_noise_regen):.3f}')
print(f'MSE между исходным сигналом и восстановленным: {mean_squared_error(x, x_DAC_noise_regen):.3f}')
print(f'R2 между исходным сигналом и восстановленным: {r2_score(x, x_DAC_noise_regen):.3f}')

plt.figure()
plt.plot(t, error_DAC_1, color='teal')
plt.title('График ошибки между исходным сигналом и восстановленным')
plt.xlabel('Время, сек')
plt.ylabel('Ошибка, e(t)')
plt.grid(True)

plt.figure()
plt.plot(t, error_DAC_2, color='tomato')
plt.title('График ошибки между исходными сигналами с и без шума')
plt.xlabel('Время, сек')
plt.ylabel('Ошибка, e(t)')
plt.grid(True)

# plt.show()

# Задание 14
# with open('T_regen.txt', 'w') as txt1:
#     txt1.write('Стат.характеристики сигнала ошибки между исходным сигналом и восстановленным: \n')

# with open('T_regen.txt', 'a') as txt1:
#     txt1.write(f'при T = {T}\t MAE = {mean_absolute_error(x, x_DAC_noise_regen):.4f}\t MSE = {mean_squared_error(x, x_DAC_noise_regen):.4f}\t R2 = {r2_score(x, x_DAC_noise_regen):.4f} \n')

# with open('T_noise.txt', 'w') as txt2:
#     txt2.write('Стат.характеристики сигнала ошибки между исходным сигналом без и с шумом: \n')

# with open('T_noise.txt', 'a') as txt2:
#     txt2.write(f'при T = {T}\tMAE = {mean_absolute_error(x, x_noise):.4f}\t MSE = {mean_squared_error(x, x_noise):.4f}\t R2 = {r2_score(x, x_noise):.4f} \n')

# with open('r_regen.txt', 'w') as txt1:
#     txt1.write('Стат.характеристики сигнала ошибки между исходным сигналом и восстановленным: \n')

with open('r_regen.txt', 'a') as txt1:
    txt1.write(f'при r = {r}\t MAE = {mean_absolute_error(x, x_DAC_noise_regen):.4f}\t MSE = {mean_squared_error(x, x_DAC_noise_regen):.4f}\t R2 = {r2_score(x, x_DAC_noise_regen):.4f} \n')

# with open('r_noise.txt', 'w') as txt2:
#     txt2.write('Стат.характеристики сигнала ошибки между исходным сигналом без и с шумом: \n')

with open('r_noise.txt', 'a') as txt2:
    txt2.write(f'при r = {r}\t MAE = {mean_absolute_error(x, x_noise):.4f}\t MSE = {mean_squared_error(x, x_noise):.4f}\t R2 = {r2_score(x, x_noise):.4f} \n')


# T_1 = 0.001
# t_1 = np.linspace(0, T_1 * N, N)
# x_1 = np.zeros_like(t_1)
# T_2 = 0.00001
# t_2 = np.linspace(0, T_2 * N, N)
# x_2 = np.zeros_like(t_2)
#
# for freq in f:
#     x_1 += np.sin(2 * np.pi * freq * t_1)
#
# for freq in f:
#     x_2 += np.sin(2 * np.pi * freq * t_2)
#
# x_1 = Amp * x / len(f)
# x_2 = Amp * x / len(f)
#
# c_ACD_1 = (U_min + (x_1 - X_min)*(U_max - U_min) / (X_max - X_min))
# c_ACD_2 = (U_min + (x_2 - X_min)*(U_max - U_min) / (X_max - X_min))
#
# x_1_noise = x_1 + noise  # сигнал с шумом
# c_ACD_noise_1 = c_ACD_1 + noise
# x_2_noise = x_2 + noise  # сигнал с шумом
# c_ACD_noise_2 = c_ACD_2 + noise
# c_ACD_noise_fft_1 = np.fft.fft(c_ACD_noise_1)
# c_ACD_noise_fft_2 = np.fft.fft(c_ACD_noise_2)
# F_1 = np.where(2 * np.abs(c_ACD_noise_fft_1) / N > L, 1, 0)
# F_2 = np.where(2 * np.abs(c_ACD_noise_fft_2) / N > L, 1, 0)
# c_filt_1 = np.fft.fft(c_ACD_noise_1)
# c_ACD_noise_regen_1 = np.real(np.fft.ifft(c_filt_1 * F_1))
# c_filt_2 = np.fft.fft(c_ACD_noise_2)
# c_ACD_noise_regen_2 = np.real(np.fft.ifft(c_filt_2 * F_2))
# x_DAC_noise_regen_1 = U_min + (c_ACD_noise_regen_1 - X_min) / (U_max - U_min) * (X_max - X_min)
# x_DAC_noise_regen_2 = U_min + (c_ACD_noise_regen_2 - X_min) / (U_max - U_min) * (X_max - X_min)
#
#
# data = [["MAE между исх.сигналом с и без шума", f"{mean_absolute_error(x, x_noise):.5f}", f"{mean_absolute_error(x_1, x_1_noise):.5f}", f"{mean_absolute_error(x_2, x_2_noise):.5f}"],
#         ["MSE между исх.сигналом с и без шума", f"{mean_squared_error(x, x_noise):.5f}", f"{mean_squared_error(x_1, x_1_noise):.5f}", f"{mean_squared_error(x_2, x_2_noise):.5f}"],
#         ["R2 между исх.сигналом с и без шума", f"{r2_score(x, x_noise):.5f}", f"{r2_score(x_1, x_1_noise):.5f}", f"{r2_score(x_2, x_2_noise):.5f}"],
#         ["MAE между исх.сигналом и восстановл.ЦАП", f"{mean_absolute_error(x, x_DAC_noise_regen):.5f}", f"{mean_absolute_error(x_1, x_DAC_noise_regen_1):.5f}", f"{mean_absolute_error(x_2, x_DAC_noise_regen_2):.5f}"],
#         ["MSE между исх.сигналом и восстановл.ЦАП", f"{mean_squared_error(x, x_DAC_noise_regen):.5f}", f"{mean_squared_error(x_1, x_DAC_noise_regen_1):.5f}", f"{mean_squared_error(x_2, x_DAC_noise_regen_2):.5f}"],
#         ["R2 между исх.сигналом и восстановл.ЦАП", f"{r2_score(x, x_DAC_noise_regen):.5f}", f"{r2_score(x_1, x_DAC_noise_regen_1):.5f}", f"{r2_score(x_2, x_DAC_noise_regen_2):.5f}"]]
#
# fig, ax = plt.subplots()
# ax.axis('off')
# table = ax.table(cellText=data, colLabels=["Метрики", f"при T={T}", f"при T={T_1}", f"при T={T_2:.5f}"], loc='center')
# table.auto_set_font_size(False)
#
# for column in range(len(data)):
#     table.auto_set_column_width(column)
#
# table.set_fontsize(11)
# table.scale(1, 2)
# plt.show()


