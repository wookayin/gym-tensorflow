'''
Copyright (c) 2018 Uber Technologies, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import numpy as np
import os

import tensorflow as tf

gym_tensorflow_module = tf.load_op_library(os.path.join(os.path.dirname(__file__), 'gym_tensorflow.so'))


class TensorFlowEnv(object):
    pass


class PythonEnv(TensorFlowEnv):
    def step(self, action, indices=None, name=None):
        if indices is None:
            indices = tf.convert_to_tensor(np.arange(self.batch_size, dtype=np.int32))
        with tf.variable_scope(name, default_name='PythonStep'):
            reward, done = tf.py_func(self._step, [action, indices], [tf.float32, tf.bool])
            reward.set_shape(indices.get_shape())
            done.set_shape(indices.get_shape())
            return reward, done

    def _reset(self, indices):
        raise NotImplementedError()

    def reset(self, indices=None, max_frames=None, name=None):
        if indices is None:
            indices = tf.convert_to_tensor(np.arange(self.batch_size, dtype=np.int32))
        with tf.variable_scope(name, default_name='PythonReset'):
            return tf.py_func(self._reset, [indices], tf.int64).op

    def _step(self, action, indices):
        raise NotImplementedError()

    def _obs(self, indices):
        raise NotImplementedError()

    def observation(self, indices=None, name=None):
        if indices is None:
            indices = tf.convert_to_tensor(np.arange(self.batch_size, dtype=np.int32))
        with tf.variable_scope(name, default_name='PythonObservation'):
            obs = tf.py_func(self._obs, [indices], tf.float32)
            obs.set_shape(tuple(indices.get_shape()) + self.observation_space[1:])
            return obs

    def final_state(self, indices, name=None):
        with tf.variable_scope(name, default_name='PythonFinalState'):
            return tf.zeros([tf.shape(indices)[0], 2], dtype=tf.float32)

    @property
    def unwrapped(self):
        return self

    def close(self):
        pass


class GymEnvWrapper(PythonEnv):
    def __init__(self, envs):
        self.env = envs
        self.batch_size = len(envs)
        self.obs = [None] * len(envs)

    @property
    def action_space(self):
        return self.env[0].action_space.n

    @property
    def observation_space(self):
        return (self.batch_size,) + self.env[0].observation_space.shape[:-1] + (1,)

    @property
    def discrete_action(self):
        return True

    def _step(self, action, indices):
        results = map(lambda i: self.env[indices[i]].step(action[i]), range(len(indices)))
        obs, reward, done, _ = zip(*results)
        for i in range(len(indices)):
            self.obs[indices[i]] = obs[i].astype(np.float32)

        return np.array(reward, dtype=np.float32), np.array(done, dtype=np.bool)

    def _reset(self, indices):
        for i in indices:
            self.obs[i] = self.env[i].reset().astype(np.float32)
        return 0

    def _obs(self, indices):
        return np.array([self.obs[i] for i in indices]).astype(np.float32)


class GymAtariWrapper(GymEnvWrapper):
    def _step(self, action, indices):
        results = map(lambda i: self.env[indices[i]].step(action[i]), range(len(indices)))
        obs, reward, done, _ = zip(*results)
        for i in range(len(indices)):
            self.obs[indices[i]] = np.expand_dims(np.dot(obs[i].astype(np.float32) * (1. / 255.), np.array([0.299, 0.587, 0.114], np.float32)), axis=-1)

        return np.array(reward, dtype=np.float32), np.array(done, dtype=np.bool)

    def _reset(self, indices):
        for i in indices:
            self.obs[i] = np.expand_dims(np.dot(self.env[i].reset().astype(np.float32) * (1. / 255.), np.array([0.299, 0.587, 0.114], np.float32)), axis=-1)
        return 0


class GymEnv(GymEnvWrapper):
    def __init__(self, name, batch_size):
        super(GymEnv, self).__init__([gym.make(name) for _ in range(batch_size)])
