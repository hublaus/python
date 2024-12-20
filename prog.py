from sudoku_package.affichage import affiche
from sudoku_package.resolution import valide, genere
from sudoku_package.gestion_fichiers import charge, sauvegarde
import sys


# pour différencier si programme appelé en main ou en package
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
    
