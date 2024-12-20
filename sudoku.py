import sys
import random

def valide(grille):
    
    #Valeurs permises dans une grille
    autorisees = range(1, 10)

    #Test des lignes
    for line in grille:
        if not len(set(line)) == 9:
            return False
        
    #Test des colonnes
    for i in range(9):
        column = []
        for j in range(9):
            column.append(grille[j][i])
        if not len(set(column)) == 9:
            return False
        
    #On teste les sous-grilles 3x3
    for y0 in [0, 3, 6]:
        for x0 in [0, 3, 6]:
            subgrid = []
            for i in range(0, 3):
                for j in range(0, 3):
                    if grille[y0+i][x0+j] in subgrid or grille[y0+i][x0+j] not in autorisees:
                        return False
                    subgrid.append(grille[y0+i][x0+j])
                    
    return True

def n_valide(y, x, n, grid):
    #Détermine si un nombre n peut être mis sur une case à la colonne x et à la ligne y
    
    #On détermine si le nombre est valide sur sa ligne
    for x0 in range(len(grid)):
        if grid[y][x0] == n:
            return False

    #On détermine si le nombre est valide sur sa colonne
    for y0 in range(len(grid)):
        if grid[y0][x] == n:
            return False
    
    x0 = (x//3) * 3
    y0= (y//3) * 3
    #On détermine si le nombre est valide dans sa sous-grille 3x3
    for i in range(0,3):
        for j in range(0,3):
            if grid[y0+i][x0+j] == n:
                return False
    return True

def genere(grid):
    
    # condition d'arrêt
    if valide(grid):
        return True
    
    for y in range(9):
        for x in range(9):
            if grid[y][x] == 0:
                r = list(range(1,10))
                random.shuffle(r)
                for n in r:
                    if n_valide(y, x, n, grid):
                        grid[y][x] = n
                        if genere(grid):
                            return True
                        grid[y][x] = 0
                return False

def affiche(M):
    m=len(M)    # nombre de lignes
    n=len(M[0]) # nombre de colonnes

    for i in range(m):
        for j in range(n):
            print('{:4}'.format(M[i][j]), end="")
        print()

def sauvegarde(grid, fichier):
    for y in range(9):
        for x in range(9):
            fichier.write(str(grid[y][x]))
            if x != 8:
                fichier.write(";")
        fichier.write("\n")

def  charge(grid, fichier):
    for y in range(9):
        ligne = fichier.readline().split(';')
        for x in range(9):
            grid[y][x] = int(ligne[x])

if __name__ == "__main__":
    
    grille = [[0 for x in range(9)] for y in range(9)]

    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

    if len(args) != 1 or len(opts) != 1:
        raise SystemExit(f"Usage: {sys.argv[0]} (-t | -g ) <fichier.csv>...")

    if "-t" in opts:
        f = open(args[0], "r")
        charge(grille, f)
        f.close()
        print("La grille ci-dessous a été chargée depuis le fichier", args[0])
        affiche(grille)
        if valide(grille):
            print("Cette grille est valide")
        else:
            print("Cette grille n'est pas valide !")

    elif "-g" in opts:
        genere(grille)
        f = open(args[0], "w")
        sauvegarde(grille, f)
        f.close()
        print("La grille ci-dessous a été sauvée dans le fichier", args[0])
        affiche(grille)
    else:
        raise SystemExit(f"Usage: {sys.argv[0]} (-t | -g ) <fichier.csv>...")



