

# Análisis de Series Temporales 101 en Python

## Introducción
Este curso tiene como objetivo proporcionarte una base para que puedas profundizar en estos modelos más adelante. Utilizaremos precios de acciones de Amazon recopilados de Yahoo Finance para hacer los modelos ARMA, ARIMA y GARCH.

## Métricas Básicas
1. **Valores P (P-Values)**: Indican la precisión de nuestro modelo. El valor debe ser menor a 0.05 para que el modelo sea considerado preciso.
2. **Coeficientes (Coefs)**: Son los números que el modelo itera para darnos el mejor resultado. Se usan para construir nuestra línea de mejor ajuste.
3. **Error Estándar (Std Err)**: Indica qué tan lejos está nuestra línea de mejor ajuste de los puntos de datos reales. Un error estándar bajo significa un modelo mejor.
4. **Sobreajuste (Over Fitting)**: Ocurre cuando el modelo funciona muy bien con datos pasados, pero mal con nuevos datos. Un espacio adecuado entre los puntos de datos y la línea de ajuste es crucial.

## Limpieza de Datos
Primero, importamos las bibliotecas necesarias y cargamos los datos de acciones de Amazon.

```python
# Importar paquetes necesarios
import numpy as np
import pandas as pd
%matplotlib inline
import yfinance as yf

# Cargar datos de acciones de Amazon
amzn_data = yf.download("AMZN", start="2010-01-01", end="2021-05-28")

# Visualizar las primeras filas
amzn_data.head()
```

Luego, recortamos el conjunto de datos desde 2015 en adelante para ver el crecimiento de las acciones.

```python
# Recortar los datos desde 2015
amzn_data = amzn_data.loc["2015-01-01":, :]
amzn_data["Close"].plot(figsize=(10,10), title='Precio de Cierre de Amazon desde 2015 hasta mayo de 2021')
```

La razón de observar el ruido y la tendencia es para mostrar la volatilidad de la acción y la dirección del precio en un periodo de tiempo determinado.

## Modelo ARMA
Un modelo ARMA está compuesto por dos componentes: AR (Auto-Regresivo) y MA (Promedio Móvil). Para preparar estos datos para un modelo ARMA, necesitamos hacer que el conjunto de datos sea estacionario aplicando un cambio porcentual.

```python
# Hacer los datos estacionarios
returns = amzn_data["Close"].pct_change().dropna()

# Importar ARMA y ajustar el modelo
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(returns, order=(5, 1, 0))
arma_result = model.fit()

# Resumen del modelo ARMA
arma_result.summary()
```

Los valores p y los coeficientes del resumen nos indican la precisión del modelo. Comparar estos valores nos ayudará a determinar si el modelo es adecuado para predecir futuros precios.

## Modelo ARIMA
El modelo ARIMA es similar al ARMA, pero incluye un componente adicional para hacer que los datos sean estacionarios. Importamos y ajustamos el modelo ARIMA de la siguiente manera:

```python
# Importar y ajustar el modelo ARIMA
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(amzn_data["Close"], order=(1, 1, 1))
arima_result = model.fit()

# Resumen del modelo ARIMA
arima_result.summary()
```

Los resultados del modelo ARIMA generalmente son más precisos que los del ARMA debido al componente adicional de diferenciación.

## Modelo GARCH
Para predecir la volatilidad, utilizamos el modelo GARCH. Continuamos usando los retornos de Amazon para construir este modelo.

```python
# Importar y ajustar el modelo GARCH
from arch import arch_model
garch = arch_model(returns, mean='Zero', vol='Garch', p=1, q=1)
garch_result = garch.fit()

# Resumen del modelo GARCH
garch_result.summary()
```

La predicción de la volatilidad nos muestra cómo puede variar el precio de la acción en los próximos días.

## Conclusión
Espero que este curso haya sido informativo y te haya ayudado a comprender mejor estos modelos. Si tienes alguna pregunta, no dudes en contactarme.

---

Este archivo puede servir como un curso práctico básico para estudiantes de maestría en Geomática, abordando los conceptos fundamentales y las implementaciones en Python de los modelos ARMA, ARIMA y GARCH.
