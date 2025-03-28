// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Particle Swarm Optimization (PSO)
//!
//! Canonical implementation of the particle swarm optimization method as outlined in \[0\] in
//! chapter II, section A.
//!
//! For details see [`ParticleSwarm`].
//!
//! ## References
//!
//! \[0\] Zambrano-Bigiarini, M. et.al. (2013): Standard Particle Swarm Optimisation 2011 at
//! CEC-2013: A baseline for future PSO improvements. 2013 IEEE Congress on Evolutionary
//! Computation. <https://doi.org/10.1109/CEC.2013.6557848>
//!
//! \[1\] <https://en.wikipedia.org/wiki/Particle_swarm_optimization>

use crate::bounds::PSOBounds;
use anyhow::{Context, Result};
use argmin::{
    argmin_error, argmin_error_closure,
    core::{ArgminFloat, CostFunction, Error, PopulationState, Problem, Solver, SyncAlias, KV},
    float,
};
use argmin_math::{ArgminRandom, ArgminSub};
use nalgebra as na;
use rand::{Rng, SeedableRng};

#[derive(Clone)]
pub struct ParticleSwarm<const N: usize, B: PSOBounds<N>, F> {
    /// Inertia weight
    pub weight_inertia: F,
    /// Cognitive acceleration coefficient
    pub weight_cognitive: F,
    /// Social acceleration coefficient
    pub weight_social: F,
    /// Delta (potential)
    pub delta: na::SVector<F, N>,
    /// Bounds on parameter space
    pub bounds: B,
    /// Number of particles
    pub num_particles: usize,
    /// Random number generator
    pub rng_generator: rand::rngs::StdRng,
    /// Initial positions, random if set to None
    pub initial_positions: Option<Vec<na::SVector<f64, N>>>,
}

impl<const N: usize, B, F> ParticleSwarm<N, B, F>
where
    B: PSOBounds<N>,
    F: ArgminFloat,
{
    pub fn new(
        bounds: B,
        num_particles: usize,
        initial_positions: Option<Vec<na::SVector<f64, N>>>,
    ) -> Self {
        ParticleSwarm {
            weight_inertia: float!(1.0f64 / (2.0 * 2.0f64.ln())),
            weight_cognitive: float!(0.5 + 2.0f64.ln()),
            weight_social: float!(0.5 + 2.0f64.ln()),
            delta: bounds.widths().map(|x| float!(x * 1e-7)),
            bounds,
            num_particles,
            rng_generator: rand::rngs::StdRng::from_entropy(),
            initial_positions,
        }
    }
}

type V<const N: usize> = na::SVector<f64, N>;

impl<const N: usize, B, F> ParticleSwarm<N, B, F>
where
    B: PSOBounds<N>,
    F: ArgminFloat,
{
    pub fn with_inertia_factor(mut self, factor: F) -> Result<Self, Error> {
        self.weight_inertia = factor;
        Ok(self)
    }

    pub fn with_cognitive_factor(mut self, factor: F) -> Result<Self, Error> {
        self.weight_cognitive = factor;
        Ok(self)
    }

    pub fn with_social_factor(mut self, factor: F) -> Result<Self, Error> {
        if factor < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`ParticleSwarm`: social factor must be >=0."
            ));
        }
        self.weight_social = factor;
        Ok(self)
    }

    // In the future we may want to allow for different delta values for each dimension
    pub fn with_delta(mut self, delta: F) -> Result<Self, Error> {
        if delta < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`ParticleSwarm`: delta must be >=0."
            ));
        }
        self.delta = self.bounds.widths().map(|x| float!(x) * delta);
        Ok(self)
    }

    /// Initializes all particles randomly and sorts them by their cost function values
    fn initialize_particles<O: CostFunction<Param = V<N>, Output = F> + SyncAlias>(
        &mut self,
        problem: &mut Problem<O>,
    ) -> Result<Vec<Particle<V<N>, F>>, Error> {
        let (positions, velocities) = self.initialize_positions_and_velocities();

        let costs = problem.bulk_cost(&positions)?;

        let mut particles = positions
            .into_iter()
            .zip(velocities)
            .zip(costs)
            .map(|((p, v), c)| Particle::new(p, c, v))
            .collect::<Vec<_>>();

        // sort them, such that the first one is the best one
        particles.sort_by(|a, b| {
            a.cost
                .partial_cmp(&b.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(particles)
    }

    /// Initializes positions and velocities for all particles
    fn initialize_positions_and_velocities(&mut self) -> (Vec<V<N>>, Vec<V<N>>) {
        if let Some(positions) = &self.initial_positions {
            let velocities = (0..self.num_particles).map(|_| V::<N>::zeros()).collect();
            (positions.clone(), velocities)
        } else {
            let (min, max) = self.bounds.outer_limits();
            let delta = max.sub(&min);
            let delta_neg = -delta;
            let positions = self
                .bounds
                .generate_random_within_bounds(&mut self.rng_generator, self.num_particles)
                .into_iter()
                .collect();
            let velocities = (0..self.num_particles)
                .map(|_| V::<N>::rand_from_range(&delta_neg, &delta, &mut self.rng_generator))
                .collect();
            (positions, velocities)
        }
    }
}

fn random_uniform(rng: &mut impl Rng, a: f64, b: f64) -> f64 {
    if a == b {
        a
    } else if a < b {
        rng.gen_range(a..b)
    } else {
        rng.gen_range(b..a)
    }
}

fn calculate_potential<const N: usize>(
    particle: &Particle<na::SVector<f64, N>, f64>,
    global_best: f64,
    dim: usize,
) -> f64 {
    particle.velocity[dim].abs() + (global_best - particle.position[dim]).abs()
}

impl<const N: usize, B, O> Solver<O, PopulationState<Particle<V<N>, f64>, f64>>
    for ParticleSwarm<N, B, f64>
where
    B: PSOBounds<N>,
    O: CostFunction<Param = V<N>, Output = f64> + SyncAlias,
    f64: ArgminFloat,
{
    const NAME: &'static str = "ParticleSwarm";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        mut state: PopulationState<Particle<V<N>, f64>, f64>,
    ) -> Result<(PopulationState<Particle<V<N>, f64>, f64>, Option<KV>), Error> {
        // Users can provide a population or it will be randomly created.
        let particles = match state.take_population() {
            Some(mut particles) if particles.len() == self.num_particles => {
                // sort them first
                particles.sort_by(|a, b| {
                    a.cost
                        .partial_cmp(&b.cost)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                particles
            }
            Some(particles) => {
                return Err(argmin_error!(
                    InvalidParameter,
                    format!(
                        "`ParticleSwarm`: Provided list of particles is of length {}, expected {}",
                        particles.len(),
                        self.num_particles
                    )
                ))
            }
            None => self.initialize_particles(problem)?,
        };

        Ok((
            state
                .individual(particles[0].clone())
                .cost(particles[0].cost)
                .population(particles),
            None,
        ))
    }

    /// Perform one iteration of algorithm
    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: PopulationState<Particle<V<N>, f64>, f64>,
    ) -> Result<(PopulationState<Particle<V<N>, f64>, f64>, Option<KV>), Error> {
        let mut best_particle = state.take_individual().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`ParticleSwarm`: No current best individual in state."
        ))?;
        let mut best_cost = state.get_cost();
        let mut particles = state.take_population().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`ParticleSwarm`: No population in state."
        ))?;

        for p in 0..particles.len() {
            // println!("Iterate particle {}: {}", p, particles[p].position);
            for d in 0..N {
                // let condition = particles.iter().all(|particle| {
                //     calculate_potential(particle, best_particle.position[d], d) < self.delta[d]
                // });
                // if condition {
                //     // forced update
                //     // println!("Forced update in d={}, i={}, p={}", d, state.iter, p);
                //     // particles[p].velocity[d] =
                //     //     random_uniform(&mut self.rng_generator, -self.delta[d], self.delta[d]);
                // } else {
                // Regular PSO update
                let momentum = particles[p].velocity[d] * self.weight_inertia;

                let to_optimum = particles[p].best_position[d] - particles[p].position[d];
                let pull_to_optimum = random_uniform(&mut self.rng_generator, 0.0, to_optimum)
                    * self.weight_cognitive;

                let to_global_optimum = best_particle.position[d] - particles[p].position[d];
                let pull_to_global_optimum =
                    random_uniform(&mut self.rng_generator, 0.0, to_global_optimum)
                        * self.weight_social;

                particles[p].velocity[d] = momentum + pull_to_optimum + pull_to_global_optimum;
                let mut new_position = particles[p].position;
                new_position[d] += particles[p].velocity[d];
                particles[p].position[d] = self
                    .bounds
                    .clamp_1d(new_position, d)
                    .with_context(|| format!("Error clamping {:?} dim {}", new_position, d))?;
                // println!("Particle {}: dim {} at {}", p, d, particles[p].position[d]);
                // }
                if !self.bounds.is_within_bounds(particles[p].position) {
                    return Err(argmin_error!(
                            InvalidParameter,
                            format!(
                                "`ParticleSwarm`: Particle {} is not within bounds: {:?}. Before clamping in {}: {:?}",
                                p, particles[p].position, d, new_position
                            )
                        ));
                }
            }
            // println!("Done particle {}: {}", p, particles[p].position);
        }
        let positions: Vec<V<N>> = particles.iter().map(|p| p.position).collect();

        let costs = problem.bulk_cost(&positions)?;

        for (p, c) in particles.iter_mut().zip(costs.into_iter()) {
            p.cost = c;

            if p.cost < p.best_cost {
                p.best_position = p.position;
                p.best_cost = p.cost;

                if p.cost < best_cost {
                    best_particle.position = p.position;
                    best_particle.best_position = p.position;
                    best_particle.cost = p.cost;
                    best_particle.best_cost = p.cost;
                    best_cost = p.cost;
                }
            }
        }

        Ok((
            state
                .individual(best_particle)
                .cost(best_cost)
                .population(particles),
            None,
        ))
    }

    // fn terminate(
    //     &mut self,
    //     state: &PopulationState<Particle<V<N>, f64>, f64>,
    // ) -> argmin::core::TerminationStatus {
    //     let particles = state.get_population().unwrap();
    //     let global_best = match state.best_individual.as_ref() {
    //         Some(p) => p.position,
    //         None => return argmin::core::TerminationStatus::NotTerminated,
    //     };
    //     let condition = particles.iter().all(|p| {
    //         for d in 0..N {
    //             if p.position[d] - global_best[d] > self.delta[d] {
    //                 return false;
    //             }
    //         }
    //         true
    //     });
    //     if condition {
    //         argmin::core::TerminationStatus::Terminated(
    //             argmin::core::TerminationReason::SolverConverged,
    //         )
    //     } else {
    //         argmin::core::TerminationStatus::NotTerminated
    //     }
    // }
}

/// A single particle
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Particle<T, F> {
    /// Position of particle
    pub position: T,
    /// Velocity of particle
    pub velocity: T,
    /// Cost of particle
    pub cost: F,
    /// Best position of particle so far
    pub best_position: T,
    /// Best cost of particle so far
    pub best_cost: F,
}

impl<T, F> Particle<T, F>
where
    T: Clone,
    F: ArgminFloat,
{
    pub fn new(position: T, cost: F, velocity: T) -> Particle<T, F> {
        Particle {
            position: position.clone(),
            velocity,
            cost,
            best_position: position,
            best_cost: cost,
        }
    }
}
