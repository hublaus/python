import cv2
import numpy as np
from tensorflow.keras.models import load_model

class DigitRecognizer:
    def __init__(self, model_path):
        """
        Initialise le modèle Keras pré-entraîné.
        """
        self.model = load_model(model_path)

    def preprocess_cell(self, cell):
        """
        Prétraite une cellule pour la rendre compatible avec le modèle (28x28, échelle [0, 1]).
        """
        # Convertir en niveaux de gris si nécessaire
        if len(cell.shape) == 3:
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

        # Binariser l'image
        _, cell = cv2.threshold(cell, 128, 255, cv2.THRESH_BINARY_INV)

        # Redimensionner à 28x28 pixels
        cell = cv2.resize(cell, (28, 28), interpolation=cv2.INTER_AREA)

        # Normaliser les valeurs entre 0 et 1
        cell = cell.astype('float32') / 255.0

        # Ajouter une dimension supplémentaire pour le modèle Keras (batch_size, 28, 28, 1)
        cell = np.expand_dims(cell, axis=(0, -1))
        return cell

    def recognize_digit(self, cell):
        """
        Reconnaît le chiffre dans une cellule donnée.
        """
        preprocessed = self.preprocess_cell(cell)
        prediction = self.model.predict(preprocessed)
        return np.argmax(prediction)  # Retourne la classe prédite (0-9)

class ImageProcessor:
    def __init__(self, image_path):
        """
        Initialise l'instance avec le chemin vers l'image.
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image introuvable : {image_path}")
        self.processed_image = None
        self.grid_image = None

    def preprocess_image(self):
        """
        Convertit l'image en niveaux de gris et applique un traitement pour détecter les contours.
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        return edged

    def find_grid_contour(self, edged):
        """
        Trouve le plus grand contour carré qui correspond à la grille du Sudoku.
        """
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximons le contour pour obtenir une forme carrée
        perimeter = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)

        #affiche traits rouges sur contours
        cv2.drawContours(self.image, [approx], -1, (0, 0, 255), 2)
        cv2.imshow("Contours", self.image)

        if len(approx) == 4:  # Si le contour est approximativement un quadrilatère
            #affiche traits rouges sur contours
            cv2.drawContours(self.image, [approx], -1, (0, 0, 255), 2)
            cv2.imshow("Contours", self.image)
            return approx
        else:
            raise ValueError("Impossible de trouver une grille carrée.")

    def warp_grid(self, grid_contour):
        """
        Redresse la perspective pour obtenir une image rectifiée de la grille.
        """
        points = grid_contour.reshape(4, 2)
        points = sorted(points, key=lambda x: (x[1], x[0]))  # Trier par coordonnée Y, puis X
        top_left, top_right = sorted(points[:2], key=lambda x: x[0])
        bottom_left, bottom_right = sorted(points[2:], key=lambda x: x[0])

        # Définir la taille de la grille rectifiée
        grid_size = 450  # Par exemple, 450x450 pixels
        destination = np.array([
            [0, 0],
            [grid_size - 1, 0],
            [0, grid_size - 1],
            [grid_size - 1, grid_size - 1]
        ], dtype="float32")

        # Calculer la matrice de transformation
        matrix = cv2.getPerspectiveTransform(np.array([top_left, top_right, bottom_left, bottom_right], dtype="float32"), destination)
        self.grid_image = cv2.warpPerspective(self.image, matrix, (grid_size, grid_size))
        return self.grid_image

    def extract_cells(self):
        """
        Découpe l'image rectifiée de la grille en 81 cellules (9x9).
        """
        if self.grid_image is None:
            raise ValueError("La grille n'a pas encore été extraite.")
        
        cells = []
        grid_size = self.grid_image.shape[0]
        cell_size = grid_size // 9

        for row in range(9):
            for col in range(9):
                x1, y1 = col * cell_size, row * cell_size
                x2, y2 = (col + 1) * cell_size, (row + 1) * cell_size
                cell = self.grid_image[y1:y2, x1:x2]
                cells.append(cell)

                # Dessiner contours des cellules en vert
                cv2.rectangle(self.grid_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

        return cells

    def overlay_solution(self, solved_grid):
        """
        Superpose la solution sur l'image originale.
        """
        # TODO : Implémenter la superposition des chiffres résolus sur l'image originale
        pass

# Exemple d'utilisation
if __name__ == "__main__":
    processor = ImageProcessor("sudoku.jpg")
    edged = processor.preprocess_image()
    contour = processor.find_grid_contour(edged)
    grid = processor.warp_grid(contour)
    cells = processor.extract_cells()

    # TODO : Résoudre chaque cellule et superposer la solution sur l'image originale
        

    # Affichage de l'image rectifiée
    cv2.imshow("Grid", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
