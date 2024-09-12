package data

// Función para dividir los datos
func splitData(data [][]float64, labels []float64, feature int, threshold float64) ([][]float64, [][]float64, []float64, []float64) {
	var leftData, rightData [][]float64
	var leftLabels, rightLabels []float64

	for i, point := range data {
		if point[feature] <= threshold {
			leftData = append(leftData, point)
			leftLabels = append(leftLabels, labels[i])
		} else {
			rightData = append(rightData, point)
			rightLabels = append(rightLabels, labels[i])
		}
	}

	return leftData, rightData, leftLabels, rightLabels
}
