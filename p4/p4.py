import pulp
import numpy as np

def resolver_rejilla_magica(N, M, L, K):
    # Crear el problema
    problema = pulp.LpProblem("Rejilla_Magica", pulp.LpMinimize)

    # Variables: una por cada celda (i,j), enteras y >= 0
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(N) for j in range(N)),
                              lowBound=0, cat='Integer')

    # No hay función objetivo (solo buscamos factibilidad), así que minimizamos 0
    problema += 0

    # Restricciones: subrejillas MxL deben sumar K
    for i in range(N - M + 1):
        for j in range(N - L + 1):
            problema += pulp.lpSum(x[i + mi, j + lj] for mi in range(M) for lj in range(L)) == K

    # Restricciones: subrejillas LxM deben sumar K
    for i in range(N - L + 1):
        for j in range(N - M + 1):
            problema += pulp.lpSum(x[i + li, j + mj] for li in range(L) for mj in range(M)) == K

    # Resolver
    problema.solve()

    # Mostrar solución
    if problema.status == pulp.LpStatusOptimal:
        rejilla = np.zeros((N, N), dtype=int)
        for i in range(N):
            for j in range(N):
                rejilla[i, j] = int(pulp.value(x[i, j]))
        print(rejilla)
    else:
        print("No se encontró solución.")

# Ejemplo de uso:
resolver_rejilla_magica(N=6, M=3, L=2, K=7)