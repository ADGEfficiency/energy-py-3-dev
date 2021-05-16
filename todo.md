Where to parallelize:
- entire experiment first (ie search over hyperparams)
- within experiment for later

Where in the code to parallelize:

- maybe nice to have an entrypoints bit?

```
energypy.parallelize([hyp1, hyp2])
```

---

last 100 train / test / random rewards

---


Can I start to split based on sampling v labelling v fitting?
- test & fill buffer should be one function
- what things read & save to disk

---

want a way to restart easily from a checkpoint
- automatically load latest
- tests for loading checkpoints

should just fill buffer if you can't find it

5 min versus 30 min
- should read from interval data

want to include env run time in counters

save stuff in checkpoint so that we can figure out how long this specific checkpoint was

include dataset creation hyper parameters in hyp.json



---

lessons
- random inital charge to get more behaviour
- reward scale very important


---

- json logging
- {"thread":"main","level":"INFO","loggerName":"mainLogger","message":{"foo":"bar"},"endOfBatch":false,"loggerFqcn":"org.apache.logging.log4j.spi.AbstractLogger","instant":{"epochSecond":1548434758,"nanoOfSecond":572000000},"threadId":1,"threadPriority":5}"

nem test / train data
- work started in tests/test_nem_dataset.py

from sac import make
sac.make

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

