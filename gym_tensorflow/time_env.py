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
import time
import threading

import tensorflow as tf

import gym_tensorflow
from gym_tensorflow.wrappers import AutoResetWrapper

def main(num_actors, num_threads, game_id='pong',
         use_tfenv=True, render=False):
    '''Speed test.

    Using 16 actors * 1 threads, it should be as high as 10000 step/s.
    Using 128 actors * 16 threads (full stress test), it should be as high as 30000 steps/s
      under a 40-thread (20-CPU * 2 HT) CPU machine.
    '''
    counter = tf.Variable(0, tf.int64)

    def make_env():
        env = AutoResetWrapper(gym_tensorflow.atari.AtariEnv(game_id, num_actors))
        tf_rew, tf_done = env.step(tf.zeros((num_actors,), tf.int32))
        render_op = env.env.render() if render else tf.no_op()

        reward_op = tf_rew
        counter_op = tf.assign_add(counter, num_actors, use_locking=True)

        def _run(sess):
            sess.run([reward_op, counter_op, render_op])
        return _run

    step_op = [make_env() for _ in range(num_threads)]

    def thread_f(sess, e):
        while True:
            e(sess)
            time.sleep(0)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        threads = [threading.Thread(target=thread_f, args=(sess, e)) for e in step_op]
        for t in threads:
            t.setDaemon(True)
            t._state = 0
            t.start()

        print("main thread start")
        tstart = time.time()
        num_steps = 0
        while True:
            diff = sess.run(counter) - num_steps
            time_str = time.strftime('%Y%m%d-%H%M%S')
            print('{}  Rate: {:.0f} steps/s'.format(time_str, diff / (time.time() - tstart)), flush=True)
            tstart = time.time()
            num_steps += diff
            time.sleep(5)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-actors', default=128, type=int, help='Number of actors per thread.')
    parser.add_argument('--num-threads', default=16, type=int, help='Number of threads to run.')
    parser.add_argument('--render', action='store_true', help='If set, fetch render op as well (slower)')
    args = parser.parse_args()

    main(num_actors=args.num_actors,
         num_threads=args.num_threads,
         render=args.render)
