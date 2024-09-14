package svm

import (
	"fmt"
	"math/rand"
	"time"
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
	if sum >= 0 {
		return 1.0
	} else {
		return 0.0
	}
}

// Entrena el modelo SVM secuencialmente usando el algoritmo de margen máximo
func (svm *SVM) trainSequential(data [][]float64, labels []float64, epochs int, learningRate float64, lambda float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		for i := range data {
			prediction := svm.predict(data[i])

			// Actualización basada en el margen
			if labels[i]*prediction < 1 {
				for j := range svm.weights {
					svm.weights[j] += learningRate * (labels[i]*data[i][j] - lambda*svm.weights[j])
				}
				svm.bias += learningRate * labels[i]
			} else {
				// Regularización
				for j := range svm.weights {
					svm.weights[j] -= learningRate * lambda * svm.weights[j]
				}
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

func SVMSecuential(train [][]float64, label_train []float64, test [][]float64, label_test []float64) {
	// rand.Seed(42) // Inicializa la semilla aleatoria

	start := time.Now()

	// Datos de ejemplo
	data := train
	labels := label_train
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
	tp, _, fp, fn := confusionMatrix(predictions, labels)
	prec := precision(tp, fp)
	rec := recall(tp, fn)
	f1 := f1Score(prec, rec)

	fmt.Printf("Precision: %.2f\n", prec)
	fmt.Printf("Accuracy: %.2f\n", acc)
	fmt.Printf("F1-Score: %.2f\n", f1)
	fmt.Printf("Recall: %.2f\n", rec)

	elapsed := time.Since(start)
	fmt.Printf("Tiempo de ejecución: %s\n", elapsed)
}
