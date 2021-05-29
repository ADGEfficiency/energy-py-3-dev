## Parallelize entire experiment first (ie search over hyperparams)

## Parallelize within expt:

Can I start to split based on sampling v labelling v fitting?
- test & fill buffer should be one function
- what things read & save to disk

energypy.make('sac', n_cpu={'sampling': 2, 'labelling': 2}, n_gpu={'fitting': 1})

---
want to include env run time in counters
- want to log this once a cycle

stuff shared globally
- rewards, counters

---

why cant it be env.reset('test')

---

want to store env info
- attempted action, actual action



---

5 min versus 30 min
- should be set in hyperparams

want a way to restart easily from a checkpoint
- automatically load latest
- tests for loading checkpoints


save stuff in checkpoint so that we can figure out how long this specific checkpoint was

include dataset creation hyper parameters in hyp.json

---

lessons
- random inital charge to get more behaviour
- reward scale very important
- filling in the prices that ocuur next period with a large negative number

---

- json logging
- {"thread":"main","level":"INFO","loggerName":"mainLogger","message":{"foo":"bar"},"endOfBatch":false,"loggerFqcn":"org.apache.logging.log4j.spi.AbstractLogger","instant":{"epochSecond":1548434758,"nanoOfSecond":572000000},"threadId":1,"threadPriority":5}"

nem test / train data
- work started in tests/test_nem_dataset.py

---

tool to analyze data  / logs
- import / export, reward

when are you ready to move over to energypy repo?
- data on s3
- thats about it :)

https://eprints.whiterose.ac.uk/159354/1/final_submitted_energy_storage_arbitrage_using_DRL%20%286%29.pdf
- their dataset
- dueling dqn

how i can improve
- transfer learning from off-policy learning of mlp data
- SAC - finer control, understand shape of the space
- different dataset

