Desempeño relativo de sedes educativas en Saber 11

**Herramienta de apoyo a la decisión para Secretarías de Educación**

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-red)](https://tu-usuario-tu-repo.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Descripción

En Colombia, los resultados de Saber 11 reflejan diferencias importantes entre
establecimientos educativos, en un contexto marcado por brechas socioeconómicas,
territoriales y de acceso a oportunidades. Comparar colegios únicamente por sus
puntajes puede ser injusto: no todos atienden estudiantes bajo las mismas
condiciones.

Este proyecto construye una **aplicación web desplegada en la nube** que permite
a las **Secretarías de Educación de las Entidades Territoriales Certificadas**
identificar:

1. **Sedes con desempeño mejor o peor de lo esperado** según su contexto
   socioeconómico, territorial e institucional.
2. **Sedes con riesgo de deterioro** en la siguiente cohorte.
3. **Sedes referentes** que obtienen resultados destacables en contextos adversos
   y que podrían estudiarse como casos de buenas prácticas.

La app está diseñada para un **usuario real no técnico** (equipos de calidad
educativa, planeación y analítica), con foco en tres principios: **contexto
claro**, **acciones derivadas** y **transparencia del modelo**.

---

## Problema y justificación

### Problema

Comparar sedes únicamente por puntajes absolutos oculta tanto establecimientos
que obtienen resultados destacables en contextos adversos, como sedes cuyo
desempeño está por debajo de lo esperable dadas sus condiciones. Además, sin
señales tempranas, las Secretarías detectan los deterioros cuando ya son
evidentes en los resultados.

### ¿Por qué Machine Learning?

Un tablero tradicional podría mostrar qué sedes obtuvieron mejores o peores
resultados, pero no estimaría **qué desempeño sería razonable esperar** dadas
sus condiciones ni permitiría **anticipar deterioros futuros** a partir de
múltiples patrones simultáneos.

Se abordan dos tareas complementarias:

| Tarea | Objetivo | Tipo | Modelos comparados |
|---|---|---|---|
| **Regresión** | Estimar el puntaje global esperado de una sede según su contexto | Supervisado · continuo | Regresión Lineal, SVR, CatBoost |
| **Clasificación** | Detectar sedes con riesgo de deterioro significativo en la siguiente cohorte | Supervisado · binario | Regresión Logística, SVM Lineal, Gradient Boosting |

### Unidad de análisis

**Sede-jornada-cohorte**: una sede educativa específica, en una jornada
determinada, para el grupo de estudiantes que presenta Saber 11 en el mismo
periodo. No se hacen predicciones sobre estudiantes individuales.