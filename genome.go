package eaopt

import "math/rand"

// A Genome is an entity that can have any number and kinds of properties. It
// can be evolved as long as it can be evaluated, mutated, crossedover, and
// cloned then it can.
type Genome interface {
	Evaluate() (float64, error)
	Mutate(rng *rand.Rand)
	Crossover(genome Genome, rng *rand.Rand)
	Clone() Genome
}

// A GenomeMLP is the same as Genome but with Specific genetic operators for EA-MLP
type GenomeMLP interface {
	AddLayer(rng *rand.Rand)
	RemoveLayer(rng *rand.Rand)
	SubstituteNeuron(rng *rand.Rand)
	Train(rng *rand.Rand)
}
