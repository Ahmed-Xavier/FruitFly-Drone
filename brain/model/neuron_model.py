"""
neuron_model.py
---------------
PyTorch implementation of the Drosophila LIF neuron model.

Implements:
  PoissonSpikeGenerator  — converts firing rates to Bernoulli spikes
  AlphaSynapse           — alpha-function synapse with configurable delay
  LIFNeuron              — leaky integrate-and-fire with surrogate ATan gradient
  AlphaLIF               — combined synapse + neuron + refractory period
  TorchModel             — full network: Poisson input + connectome + AlphaLIF

Source: run_pytorch.py from fly-brain-full (Rojas Aliaga 2026),
        originally based on Shiu et al. (2024) and Kakaria & de Bivort (2017).
"""

import numpy as np
import torch
import torch.nn as nn


class PoissonSpikeGenerator(nn.Module):
    """Converts firing rates (Hz) to Bernoulli spike trains for one timestep."""

    def __init__(self, dt, scale, device='cpu'):
        super().__init__()
        self.prob_scale = dt / 1000.0  # dt in ms → probability per step
        self.scale = scale
        self.device = device

    def forward(self, rates, generator=None):
        return torch.bernoulli(rates * self.prob_scale, generator=generator) * self.scale


class AlphaSynapse(nn.Module):
    """Alpha-function synapse dynamics with configurable axonal delay."""

    def __init__(self, batch, size, dt, params, device='cpu'):
        super().__init__()
        self.time_factor = dt / params['tauSyn']
        self.steps_delay = int(params['tDelay'] / dt)
        self.size = size
        self.device = device
        self.batch = batch

    def state_init(self):
        conductance = torch.zeros(self.batch, self.size, device=self.device)
        delay_buffer = torch.zeros(
            self.batch, self.steps_delay + 1, self.size, device=self.device
        )
        return conductance, delay_buffer

    def forward(self, input_, conductance, delay_buffer, refrac):
        conductance_new = (
            conductance * (1 - self.time_factor) + delay_buffer[:, 0, :] * refrac
        )
        delay_buffer = torch.roll(delay_buffer, shifts=-1, dims=1)
        delay_buffer[:, -1, :] = input_
        return conductance_new, delay_buffer


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with surrogate ATan gradient."""

    def __init__(self, batch, size, dt, params, device='cpu'):
        super().__init__()
        self.size = size
        self.dt = dt
        self.tau_mem = params['tauMem']
        self.v_reset = params['vReset']
        self.v_rest = params['vRest']
        self.v_threshold = params['vThreshold']
        self.v_0 = params['v0']
        self.time_factor = dt / self.tau_mem
        self.spike_gradient = self.ATan.apply
        self.device = device
        self.batch = batch

    def state_init(self):
        v = torch.zeros(self.batch, self.size, device=self.device) + self.v_0
        spikes = torch.zeros(self.batch, self.size, device=self.device)
        return spikes, v

    def forward(self, input_current, v):
        v = v + self.time_factor * (input_current - (v - self.v_rest))
        spike = self.spike_gradient(v - self.v_threshold)
        reset = ((v - self.v_reset) * spike).detach()
        v = v - reset
        return spike, v

    @staticmethod
    class ATan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, v):
            spike = (v > 0).float()
            ctx.save_for_backward(v)
            return spike

        @staticmethod
        def backward(ctx, grad_output):
            (v,) = ctx.saved_tensors
            grad = 1 / (1 + (np.pi * v).pow_(2)) * grad_output
            return grad


class AlphaLIF(nn.Module):
    """LIF neuron with alpha synapse and refractory period — one population."""

    def __init__(self, batch, size, dt, params, device='cpu'):
        super().__init__()
        self.size = size
        self.synapse = AlphaSynapse(batch, size, dt, params, device=device)
        self.neuron = LIFNeuron(batch, size, dt, params, device=device)
        self.steps_refrac = int(params['tRefrac'] / dt)

    def state_init(self):
        conductance, delay_buffer = self.synapse.state_init()
        spikes, v = self.neuron.state_init()
        refrac = self.steps_refrac + torch.zeros_like(v)
        return conductance, delay_buffer, spikes, v, refrac

    def forward(self, input_, conductance, delay_buffer, spikes, v, refrac):
        refrac = refrac * (1 - spikes) + 1
        conductance_new, delay_buffer = self.synapse(
            input_, conductance, delay_buffer, (refrac > self.steps_refrac).float()
        )
        spikes, v_new = self.neuron(conductance, v)
        conductance_reset = (conductance_new * spikes).detach()
        conductance_new = conductance_new - conductance_reset
        return conductance_new, delay_buffer, spikes, v_new, refrac


class TorchModel(nn.Module):
    """
    Full connectome model: Poisson spike injection + recurrent weights + AlphaLIF.

    weights must be a sparse tensor (CSR or COO) built from the FlyWire
    connectivity parquet file.
    """

    def __init__(self, batch, size, dt, params, weights, device='cpu'):
        super().__init__()
        self.neurons = AlphaLIF(batch, size, dt, params, device=device)
        self.weights = weights
        self.poisson = PoissonSpikeGenerator(dt, params['scalePoisson'], device=device)
        self.scale = params['wScale']

    def state_init(self):
        return self.neurons.state_init()

    def forward(self, rates, conductance, delay_buffer, spikes, v, refrac, generator=None):
        spikes_input = self.poisson(rates, generator=generator)
        weighted_spikes = torch.matmul(spikes, self.weights.transpose(0, 1))
        conductance, delay_buffer, spikes, v, refrac = self.neurons(
            self.scale * (spikes_input + weighted_spikes),
            conductance, delay_buffer, spikes, v, refrac,
        )
        return conductance, delay_buffer, spikes, v, refrac
