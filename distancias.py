import numpy as np

def levenshtein_matriz(x, y, threshold=None):
    # esta versión no utiliza threshold, se pone porque se puede
    # invocar con él, en cuyo caso se ignora
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int32)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
            )
    return D[lenX, lenY]
    

def levenshtein_edicion(x, y, threshold=None):
    # a partir de la versión levenshtein_matriz

    #calculamos matriz de edición
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int32)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
            )

    #recuperamos el camino seguido
    camino = []

    # Recorremos la matriz en sentido inverso
    while i > 0 or j > 0:
        if i > 0 and D[i][j] == D[i-1][j] + 1:
            # Operación de eliminación
            camino.append((x[i-1], ''))  # Eliminación de x[i-1]
            i -= 1
        elif j > 0 and D[i][j] == D[i][j-1] + 1:
            # Operación de inserción
            camino.append(('', y[j-1]))  # Inserción de y[j-1]
            j -= 1
        else:
            # Operación de sustitución o coincidencia
            if x[i-1] != y[j-1]:
                camino.append((x[i-1], y[j-1]))  # Sustitución de x[i-1] por y[j-1]
            else:
                camino.append((x[i-1], x[i-1]))  # Coincidencia, no hay cambio
            i -= 1
            j -= 1

    # Invertimos el camino para que esté en orden desde el inicio
    camino.reverse()

    return int(D[lenX, lenY]), camino

#levenshtein menos coste espacial no theshold
def levenshtein_reduccion(x, y, threshold=None):
    toomuch = False #si se pasa del threshold
    lenX, lenY = len(x), len(y) #longitud de las cadenas
    current_row = np.zeros(lenX + 1, dtype=int)  # La fila actual (longitud de la primera palabra + 1)
    previous_row = np.zeros(lenX + 1, dtype=int)  # La fila previa (longitud de la primera palabra + 1)
    current_row[0] = 0 

    for i in range(1, lenX + 1): #desde el principio de la primera palabra hasta su final
        current_row[i] = current_row[i - 1] + 1 #la columna actual es es coste de la columna anterior + 1
    
    for j in range(1, lenY + 1): #recorre la segunda palabra
        if toomuch: break

        previous_row, current_row = current_row, previous_row #la fila actual se guarda en previous y se modifica la actual (usando la previous)
        current_row[0] = previous_row[0] + 1 #inicializa el primer elemento de la nueva fila
        
        for i in range(1, lenX + 1): #añade los costes de la fila
            current_row[i] = min( #elige el mínimo entre
                current_row[i - 1] + 1, #lado izquierdo
                previous_row[i] + 1, #abajo
                previous_row[i - 1] + (x[i - 1] != y[j - 1]), #diagonal
            )

    return current_row[lenX]

#levenshtein coste espacial threshold
def levenshtein(x, y, threshold):
    lenX, lenY = len(x), len(y) #longitud de las cadenas
    current_row = np.zeros(lenX + 1, dtype=int)  # La fila actual (longitud de la primera palabra + 1)
    previous_row = np.zeros(lenX + 1, dtype=int)  # La fila previa (longitud de la primera palabra + 1)
    current_row[0] = 0 

    for i in range(1, lenX + 1): #desde el principio de la primera palabra hasta su final
        current_row[i] = current_row[i - 1] + 1 #la columna actual es es coste de la columna anterior + 1
    for j in range(1, lenY + 1): #recorre la segunda palabra
        previous_row, current_row = current_row, previous_row #la fila actual se guarda en previous y se modifica la actual (usando la previous)
        current_row[0] = previous_row[0] + 1 #inicializa el primer elemento de la nueva fila
        for i in range(1, lenX + 1): #añade los costes de la fila
            current_row[i] = min( #elige el mínimo entre
                current_row[i - 1] + 1, #lado izquierdo
                previous_row[i] + 1, #abajo
                previous_row[i - 1] + (x[i - 1] != y[j - 1]), #diagonal
            )
        if min(current_row) > threshold:  #comprueba que el mínimo no supere el threshold
            return threshold+1 #si supera el threshold lo devuelve sumándole 1
    return current_row[lenX]

def levenshtein_cota_optimista(x, y, threshold):
    
    vocab = {}

    # recorremos x sumando
    for letter in x:
        if letter not in vocab:
            vocab[letter] = 0
        vocab[letter] += 1

    # recorremos y restando
    for letter in y:
        if letter not in vocab:
            vocab[letter] = 0
        vocab[letter] -= 1

    #devolvemos el maximo valor en valor absoluto
    diferencias = list(vocab.values())

    #devolver el de mayor valor absoluto
    maxDiff = max(diferencias, key=abs)
    if maxDiff>threshold: return threshold+1
    else: return levenshtein(x, y, threshold)

def damerau_restricted_matriz(x, y, threshold=None):
    # completar versión Damerau-Levenstein restringida con matriz
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int32)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
            )
            if i >= 2 and j>=2 and x[i-2] == y[j-1] and x[i-1] == y[j-2]:
                D[i][j] = min(D[i][j],D[i-2][j-2]+1)
    return D[lenX, lenY]

def damerau_restricted_edicion(x, y, threshold=None):
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int32)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
            )
            if i >= 2 and j>=2 and x[i-2] == y[j-1] and x[i-1] == y[j-2]:
                D[i][j] = min(D[i][j],D[i-2][j-2]+1)
    #recuperamos el camino seguido
    camino = []

    # Recorremos la matriz en sentido inverso
    while i > 0 or j > 0:
        if i > 0 and D[i][j] == D[i-1][j] + 1:
            # Operación de eliminación
            camino.append((x[i-1], ''))  # Eliminación de x[i-1]
            i -= 1
        elif j > 0 and D[i][j] == D[i][j-1] + 1:
            # Operación de inserción
            camino.append(('', y[j-1]))  # Inserción de y[j-1]
            j -= 1
        else:
            # Operación de sustitución o coincidencia
            if i >= 2 and j >= 2 and x[i-2] == y[j-1] and x[i-1] == y[j-2]:
                camino.append((str(x[i-2])+str(x[i-1]), str(y[j-2])+str(y[j-1])))
                i -= 2
                j -= 2
            else:
                if x[i-1] != y[j-1]:
                    camino.append((x[i-1], y[j-1]))  # Sustitución de x[i-1] por y[j-1]
                else:
                    camino.append((x[i-1], x[i-1]))  # Coincidencia, no hay cambio
                i -= 1
                j -= 1

    # Invertimos el camino para que esté en orden desde el inicio
    camino.reverse()

    return D[lenX, lenY], camino

def damerau_restricted(x, y, threshold=None):
    # versión con reducción coste espacial y parada por threshold
    lenX, lenY = len(x), len(y)
    current_row = [None] * (1 + lenX) 
    previous_row = [None] * (1 + lenX) 
    pprevious_row = [None] * (1 + lenX) 
    current_row[0] = 0 

    for i in range(1, lenX + 1): 
        current_row[i] = current_row[i - 1] + 1
    if lenY >= 1:
        previous_row, current_row = current_row, previous_row 
        current_row[0] = previous_row[0] + 1 
        for i in range(1, lenX + 1): 
            current_row[i] = min( 
                current_row[i - 1] + 1, 
                previous_row[i] + 1,
                previous_row[i - 1] + (x[i - 1] != y[1 - 1]),
            )
    for j in range(2, lenY + 1): 
        pprevious_row, previous_row, current_row = previous_row, current_row, pprevious_row
        current_row[0] = previous_row[0] + 1 
        for i in range(1, lenX + 1):  
            current_row[i] = min( 
                current_row[i - 1] + 1, 
                previous_row[i] + 1,
                previous_row[i - 1] + (x[i - 1] != y[j - 1]),
            )
            if i >= 2 and x[i-2] == y[j-1] and x[i-1] == y[j-2]:
                current_row[i] = min(current_row[i],pprevious_row[i-2]+1)
        if min(current_row) > threshold:
            return threshold+1
    return current_row[lenX]

def damerau_intermediate_matriz(x, y, threshold=None):
    # completar versión Damerau-Levenstein intermedia con matriz
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int32)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1, #operación de borrado con coste 1
                D[i][j - 1] + 1, #operación de inserción con coste 1
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]), #operación de sustitución, coste 1 si son carácteres distintos, 0 en caso contrario
            )
            if i >= 2 and j>=2 and x[i-2] == y[j-1] and x[i-1] == y[j-2]:
                D[i][j] = min(D[i][j],D[i-2][j-2]+1) #operación de transposición con coste 1 (ab <-> ba)
            if i >= 3 and j>=2 and x[i-3] == y[j-1] and x[i-1] == y[j-2]:
                D[i][j] = min(D[i][j],D[i-3][j-2]+2) #operación de transposición con coste 2, tal que (acb <-> ba)
            if i>=2 and j >= 3 and x[i-2] == y[j-1] and x[i-1] == y[j-3]:
                D[i][j] = min(D[i][j],D[i-2][j-3]+2) #operación de transposición con coste 2, tal que (ab <-> bca)
    return D[lenX, lenY]

def damerau_intermediate_edicion(x, y, threshold=None):
    # partiendo de matrix_intermediate_damerau añadir recuperar
    # secuencia de operaciones de edición
    # completar versión Damerau-Levenstein intermedia con matriz
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int32)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
            )
            if i >= 2 and j >= 2 and x[i-2] == y[j-1] and x[i-1] == y[j-2]:
                D[i][j] = min(D[i][j],D[i-2][j-2]+1)
            if i >= 3 and j >= 2 and x[i-3] == y[j-1] and x[i-1] == y[j-2]:
                D[i][j] = min(D[i][j],D[i-3][j-2]+2)
            if i >= 2 and j >= 3 and x[i-2] == y[j-1] and x[i-1] == y[j-3]:
                D[i][j] = min(D[i][j],D[i-2][j-3]+2)

    camino = []

    #Recuperación del camino
    while i > 0 or j > 0:
        if i > 0 and D[i][j] == D[i-1][j] + 1:
            # Operación de eliminación
            camino.append((x[i-1], ''))  # Eliminación de x[i-1]
            i -= 1
        elif j > 0 and D[i][j] == D[i][j-1] + 1:
            # Operación de inserción
            camino.append(('', y[j-1]))  # Inserción de y[j-1]
            j -= 1
        else:
            #Caso transposición restringida (ab <-> ba)
            if i >= 2 and j >= 2 and x[i-2] == y[j-1] and x[i-1] == y[j-2]:
                camino.append((x[i-2:i],y[j-2:j]))
                i -= 2
                j -= 2
            #Caso transposición: (acb <-> ba)
            elif i >= 3 and j >= 2 and x[i-3] == y[j-1] and x[i-1] == y[j-2]:
                camino.append((x[i-3:i],y[j-2:j]))
                i -= 3
                j -= 2
            #Caso transposición: (ab <-> bca)
            elif i >= 2 and j >= 3 and x[i-2] == y[j-1] and x[i-1] == y[j-3]:
                camino.append((x[i-2:i],y[j-3:j]))
                i -= 2
                j -= 3
            # Operación de sustitución o coincidencia
            else:
                if x[i-1] != y[j-1]:
                    camino.append((x[i-1], y[j-1]))  # Sustitución de x[i-1] por y[j-1]
                else:
                    camino.append((x[i-1], x[i-1]))  # Coincidencia, no hay cambio
                i -= 1
                j -= 1

    # Invertimos el camino para que esté en orden desde el inicio
    camino.reverse()

    return D[lenX, lenY], camino

    
def damerau_intermediate(x, y, threshold=None):
    # versión con reducción coste espacial y parada por threshold
    #Se utilizan 4 vectores columna
    lenX, lenY = len(x), len(y)
    current_row = [None] * (1 + lenX) 
    previous_row = [None] * (1 + lenX) 
    pprevious_row = [None] * (1 + lenX)
    ppprevious_row = [None] * (1 + lenX)
    current_row[0] = 0 

    #Se inicializa el vector columna actual
    for i in range(1, lenX + 1): 
        current_row[i] = current_row[i - 1] + 1
    for j in range(1, lenY + 1):
        #Se actualiza cada vector
        ppprevious_row, pprevious_row, previous_row, current_row = pprevious_row, previous_row, current_row, ppprevious_row
            
        current_row[0] = previous_row[0] + 1 
        #Calculo de la operación de menor coste para todo el vector columna actual
        for i in range(1, lenX + 1):  
            current_row[i] = min( 
                current_row[i - 1] + 1, 
                previous_row[i] + 1,
                previous_row[i - 1] + (x[i - 1] != y[j - 1]),
            )
            if i >= 2 and j >= 2 and x[i-2] == y[j-1] and x[i-1] == y[j-2]:
                current_row[i] = min(current_row[i],pprevious_row[i-2]+1)
            if i >= 3 and j >= 2 and x[i-3] == y[j-1] and x[i-1] == y[j-2]:
                current_row[i] = min(current_row[i],pprevious_row[i-3]+2)
            if i >= 2 and j >= 3 and x[i-2] == y[j-1] and x[i-1] == y[j-3]:
                current_row[i] = min(current_row[i],ppprevious_row[i-2]+2)
        #Implementación de la parada por threshold
        if min(current_row) > threshold:
            return threshold+1
    return current_row[lenX]

opcionesSpell = {
    'levenshtein_m': levenshtein_matriz,
    'levenshtein_r': levenshtein_reduccion,
    'levenshtein':   levenshtein,
    'levenshtein_o': levenshtein_cota_optimista,
    'damerau_rm':    damerau_restricted_matriz,
    'damerau_r':     damerau_restricted,
    'damerau_im':    damerau_intermediate_matriz,
    'damerau_i':     damerau_intermediate
}

opcionesEdicion = {
    'levenshtein': levenshtein_edicion,
    'damerau_r':   damerau_restricted_edicion,
    'damerau_i':   damerau_intermediate_edicion
}

if __name__ == "__main__":
    print(damerau_intermediate_edicion("algoritmo","algortximo"))