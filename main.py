import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Definimos los rangos de las variables
# * La temperatura exterior está en el rango de [0, 40] grados
# * La temperatura interior está en el rango de [0, 150] grados
# * El tamaño de la llama está en el rango de 0 a 100% (control del tamaño de la llama)

x_ext = np.arange(0, 41, 1)  # Temperatura exterior [0, 40]
x_int = np.arange(0, 151, 1)  # Temperatura interior [0, 150]
x_flame = np.arange(0, 101, 1)  # Tamaño de la llama [0, 100]

# Definimos las funciones de pertenencia difusas para la temperatura exterior
ext_low = fuzz.trimf(x_ext, [0, 0, 15])  # Exterior baja
ext_med = fuzz.trimf(x_ext, [10, 20, 30])  # Exterior media
ext_high = fuzz.trimf(x_ext, [25, 40, 40])  # Exterior alta

# Definimos las funciones de pertenencia difusas para la temperatura interior
int_normal = fuzz.trimf(x_int, [0, 0, 90])  # Interior normal
int_high = fuzz.trimf(x_int, [80, 100, 120])  # Interior alta
int_critical = fuzz.trimf(x_int, [110, 150, 150])  # Interior crítica

# Definimos las funciones de pertenencia difusas para el tamaño de la llama
flame_low = fuzz.trimf(x_flame, [0, 0, 20])  # Llama en piloto
flame_med = fuzz.trimf(x_flame, [10, 40, 70])  # Llama moderada
flame_high = fuzz.trimf(x_flame, [60, 100, 100])  # Llama alta

# Graficamos las funciones de pertenencia para la temperatura exterior
fig, ax0 = plt.subplots(figsize=(8, 3))
ax0.plot(x_ext, ext_low, 'b', linewidth=1.5, label='Exterior Baja')
ax0.plot(x_ext, ext_med, 'g', linewidth=1.5, label='Exterior Media')
ax0.plot(x_ext, ext_high, 'r', linewidth=1.5, label='Exterior Alta')
ax0.set_title('Funciones de Pertenencia para la Temperatura Exterior')
ax0.legend()

plt.tight_layout()

# Graficamos las funciones de pertenencia para la temperatura interior
fig, ax1 = plt.subplots(figsize=(8, 3))
ax1.plot(x_int, int_normal, 'b', linewidth=1.5, label='Interior Normal')
ax1.plot(x_int, int_high, 'g', linewidth=1.5, label='Interior Alta')
ax1.plot(x_int, int_critical, 'r', linewidth=1.5, label='Interior Crítica')
ax1.set_title('Funciones de Pertenencia para la Temperatura Interior')
ax1.legend()

plt.tight_layout()

# Graficamos las funciones de pertenencia para el tamaño de la llama
fig, ax2 = plt.subplots(figsize=(8, 3))
ax2.plot(x_flame, flame_low, 'b', linewidth=1.5, label='Llama Piloto')
ax2.plot(x_flame, flame_med, 'g', linewidth=1.5, label='Llama Moderada')
ax2.plot(x_flame, flame_high, 'r', linewidth=1.5, label='Llama Alta')
ax2.set_title('Funciones de Pertenencia para el Tamaño de la Llama')
ax2.legend()

plt.tight_layout()

# Suponemos que tenemos un valor de entrada para la temperatura exterior e interior
# Temperatura exterior 10°C, temperatura interior 120°C
ext_input = 10
int_input = 80

# Fuzzificación de los valores de entrada
ext_level_low = fuzz.interp_membership(x_ext, ext_low, ext_input)
ext_level_med = fuzz.interp_membership(x_ext, ext_med, ext_input)
ext_level_high = fuzz.interp_membership(x_ext, ext_high, ext_input)

int_level_normal = fuzz.interp_membership(x_int, int_normal, int_input)
int_level_high = fuzz.interp_membership(x_int, int_high, int_input)
int_level_critical = fuzz.interp_membership(x_int, int_critical, int_input)

# Aplicamos las reglas difusas

# Regla 1: Si la temperatura exterior es baja y la interior es normal, entonces la llama es alta
rule1 = np.fmin(ext_level_low, int_level_normal)
flame_activation_high_rule1 = np.fmin(rule1, flame_high)

# Regla 2: Si la temperatura exterior es media y la interior es normal, entonces la llama es moderada
rule2 = np.fmin(ext_level_med, int_level_normal)
flame_activation_med_rule2 = np.fmin(rule2, flame_med)

# Regla 3: Si la temperatura exterior es alta y la interior es normal, entonces la llama está en piloto
rule3 = np.fmin(ext_level_high, int_level_normal)
flame_activation_low_rule3 = np.fmin(rule3, flame_low)

# Regla 4: Si la temperatura interior es alta, la llama es moderada
rule4 = int_level_high
flame_activation_med_rule4 = np.fmin(rule4, flame_med)

# Regla 5: Si la temperatura interior es crítica, la llama está en piloto
rule5 = int_level_critical
flame_activation_low_rule5 = np.fmin(rule5, flame_low)

# Agregación de las salidas
flame0 = np.zeros_like(x_flame)
aggregated = np.fmax(flame_activation_high_rule1,
                     np.fmax(flame_activation_med_rule2,
                             np.fmax(flame_activation_low_rule3,
                                     np.fmax(flame_activation_med_rule4, flame_activation_low_rule5))))

# Defuzzificación (centroide)
flame_output = fuzz.defuzz(x_flame, aggregated, 'centroid')
flame_activation = fuzz.interp_membership(x_flame, aggregated, flame_output)

# Graficamos la salida de las funciones de pertenencia y el resultado
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(x_flame, flame0, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot(x_flame, flame_low, 'b', linewidth=0.5, linestyle='--')
ax0.plot(x_flame, flame_med, 'g', linewidth=0.5, linestyle='--')
ax0.plot(x_flame, flame_high, 'r', linewidth=0.5, linestyle='--')
ax0.set_title('Agregación de las funciones de pertenencia y resultado')

# Resultado de defuzzificación
plt.plot([flame_output, flame_output], [0, flame_activation], 'k', linewidth=1.5, alpha=0.9)
plt.tight_layout()

plt.show()

print(f"El tamaño de la llama es: {flame_output:.2f}%")