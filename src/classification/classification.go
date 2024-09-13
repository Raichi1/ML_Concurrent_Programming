package classification

import (
	"math"
)

// Función para calcular la accuracy
func accuracy(tp, tn, fp, fn int) float64 {
	total := tp + tn + fp + fn
	if total == 0 {
		return 0.0
	}

	return float64(tp+tn) / float64(total)
}

// Function to calculate precision
func precision(tp, fp int) float64 {
	if tp+fp == 0 {
		return 0
	}
	return float64(tp) / float64(tp+fp)
}

// Function to calculate recall
func recall(tp, fn int) float64 {
	if tp+fn == 0 {
		return 0
	}
	return float64(tp) / float64(tp+fn)
}

// Function to calculate F1-score
func f1Score(precision, recall float64) float64 {
	if precision+recall == 0 {
		return 0
	}
	return 2 * (precision * recall) / (precision + recall)
}

// MSE calcula el error cuadrático medio
func mse(predictions []float64, actuals []float64) float64 {
	sum := 0.0
	for i := range predictions {
		diff := predictions[i] - actuals[i]
		sum += diff * diff
	}
	return sum / float64(len(actuals))
}

// RMSE calcula la raíz del error cuadrático medio
func rmse(predictions []float64, actuals []float64) float64 {
	return math.Sqrt(mse(predictions, actuals))
}
