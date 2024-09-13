package main

import (
	"fmt"
	"math/rand"
)

// Estructura del modelo SVM
type SVM struct {
	weights []float64
	bias    float64
}

// Inicializa el modelo SVM
func newSVM(inputSize int) *SVM {
	weights := make([]float64, inputSize)
	for i := range weights {
		weights[i] = rand.Float64()*2 - 1 // Valores aleatorios entre -1 y 1
	}
	return &SVM{
		weights: weights,
		bias:    rand.Float64()*2 - 1,
	}
}

// Calcula la predicción del modelo
func (svm *SVM) predict(inputs []float64) float64 {
	sum := svm.bias
	for i := range inputs {
		sum += inputs[i] * svm.weights[i]
	}
	return sum
}

// Entrena el modelo SVM secuencialmente usando el algoritmo de margen máximo
func (svm *SVM) trainSequential(data [][]float64, labels []float64, epochs int, learningRate float64, lambda float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		for i := range data {
			prediction := svm.predict(data[i])
			error := labels[i] - prediction

			// Actualizar pesos y sesgo
			if error != 0 {
				for j := range svm.weights {
					svm.weights[j] += learningRate*(labels[i]-prediction)*data[i][j] - lambda*svm.weights[j]
				}
				svm.bias += learningRate * (labels[i] - prediction)
			}
		}
	}
}

// Calcula la precisión
func accuracy(predictions, labels []float64) float64 {
	correct := 0
	total := len(labels)
	for i := range predictions {
		if (predictions[i] >= 0 && labels[i] == 1) || (predictions[i] < 0 && labels[i] == 0) {
			correct++
		}
	}
	return float64(correct) / float64(total)
}

// Calcula la matriz de confusión
func confusionMatrix(predictions, labels []float64) (int, int, int, int) {
	tp, tn, fp, fn := 0, 0, 0, 0
	for i := range predictions {
		if predictions[i] >= 0 && labels[i] == 1 {
			tp++
		} else if predictions[i] < 0 && labels[i] == 0 {
			tn++
		} else if predictions[i] >= 0 && labels[i] == 0 {
			fp++
		} else if predictions[i] < 0 && labels[i] == 1 {
			fn++
		}
	}
	return tp, tn, fp, fn
}

// Calcula la precisión (precision)
func precision(tp, fp int) float64 {
	if tp+fp == 0 {
		return 0.0
	}
	return float64(tp) / float64(tp+fp)
}

// Calcula el recall
func recall(tp, fn int) float64 {
	if tp+fn == 0 {
		return 0.0
	}
	return float64(tp) / float64(tp+fn)
}

// Calcula el F1-score
func f1Score(precision, recall float64) float64 {
	if precision+recall == 0 {
		return 0.0
	}
	return 2 * (precision * recall) / (precision + recall)
}

func main() {
	rand.Seed(42) // Inicializa la semilla aleatoria

	// Datos de ejemplo
	data := [][]float64{{2.3, 4.5}, {1.3, 3.5}, {3.3, 2.5}}
	labels := []float64{1.0, 0.0, 1.0}

	// Inicializar el modelo SVM
	svm := newSVM(len(data[0]))

	// Entrenar el modelo secuencialmente
	svm.trainSequential(data, labels, 100, 0.01, 0.001)

	// Hacer predicciones
	predictions := make([]float64, len(data))
	for i, d := range data {
		predictions[i] = svm.predict(d)
	}

	// Evaluar el modelo
	acc := accuracy(predictions, labels)
	tp, tn, fp, fn := confusionMatrix(predictions, labels)
	prec := precision(tp, fp)
	rec := recall(tp, fn)
	f1 := f1Score(prec, rec)

	fmt.Printf("Confusion Matrix:\n")
	fmt.Printf("TP: %d, TN: %d, FP: %d, FN: %d\n", tp, tn, fp, fn)
	fmt.Printf("Precision: %.2f\n", prec)
	fmt.Printf("Accuracy: %.2f\n", acc)
	fmt.Printf("F1-Score: %.2f\n", f1)
	fmt.Printf("Recall: %.2f\n", rec)
}
