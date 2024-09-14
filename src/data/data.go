package data

import (
	"errors"
	"math/rand"
	"time"
)

// Función para dividir los datos
func SplitData(data [][]float64, labels []float64, percentage float64) ([][]float64, [][]float64, []float64, []float64, error) {
	// Verificar que las longitudes coincidan
	if len(data) != len(labels) {
		return nil, nil, nil, nil, errors.New("la longitud de los datos y las etiquetas no coinciden")
	}

	// Verificar que el porcentaje esté en el rango [0.0, 1.0]
	if percentage < 0.0 || percentage > 1.0 {
		return nil, nil, nil, nil, errors.New("el porcentaje debe estar entre 0.0 y 1.0")
	}

	total := len(data)
	splitIndex := int(float64(total) * percentage)

	// Crear índices aleatorios
	rand.Seed(time.Now().UnixNano()) // Usar una semilla para garantizar aleatoriedad
	indices := rand.Perm(total)

	// Inicializar slices con la capacidad adecuada
	trainData := make([][]float64, 0, splitIndex)
	trainLabels := make([]float64, 0, splitIndex)
	testData := make([][]float64, 0, total-splitIndex)
	testLabels := make([]float64, 0, total-splitIndex)

	// Llenar los conjuntos de entrenamiento y prueba
	for i, idx := range indices {
		if i < splitIndex {
			trainData = append(trainData, data[idx])
			trainLabels = append(trainLabels, labels[idx])
		} else {
			testData = append(testData, data[idx])
			testLabels = append(testLabels, labels[idx])
		}
	}

	return trainData, testData, trainLabels, testLabels, nil
}
