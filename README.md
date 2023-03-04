# README #

This README will guide you on how to get the TweetSentBR dataset. The dataset was formed in 2017 by Henrico Brum, MsC student at University of Sao Paulo - ICMC.
Further description of the method is available in the paper "Building a Sentiment Corpus of Tweets in Brazilian Portuguese" (BRUM, H. B. and NUNES, M. G. V., 2018) presented at the 20th Language Resources and Evaluation Conference in Miyazaki, Japan on 2018.

Please, use the following bibtex citation whenever you use this corpus. It keeps me motivated for labeling more data in the future ;)

```
@InProceedings{BRUM18.389,
  author = {Henrico Brum and Maria das Gra\c{c}as Volpe Nunes},
  title = "{Building a Sentiment Corpus of Tweets in Brazilian Portuguese}",
  booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
  year = {2018},
  month = {May 7-12, 2018},
  address = {Miyazaki, Japan},
  editor = {Nicoletta Calzolari (Conference chair) and Khalid Choukri and Christopher Cieri and Thierry Declerck and Sara Goggi and Koiti Hasida and Hitoshi Isahara and Bente Maegaard and Joseph Mariani and HÚlŔne Mazo and Asuncion Moreno and Jan Odijk and Stelios Piperidis and Takenobu Tokunaga},
  publisher = {European Language Resources Association (ELRA)},
  isbn = {979-10-95546-00-9},
  language = {english}
}
```

Link to paper in [LREC Proceedings](http://www.lrec-conf.org/proceedings/lrec2018/summaries/389.html)

### What is TweetSentBR? ###

TweetSentBR is a corpus of Tweets in Brazilian Portuguese. It was labeled by several annotators following steps stablished on the literature for improving reliability on the task of Sentiment Analysis.
Each Tweet was annotated in one of the three following classes:

* Positive - tweets where a user meant a positive reaction or evaluation about the main topic on the post;

* Negative - tweets where a user meant a negative reaction or evaluation about the main topic on the post;

* Neutral - tweets not belonging to any of the last classes, usually not making a point, out of topic, irrelevant, confusing or containing only objective data.

### How do I obtain the corpus?

You only need three things:

* The script 'get_ttsbr.py';

* The [python-twitter library](https://python-twitter.readthedocs.io) - well, maybe you will need more libraries... I am not sure;

* A valid [Twitter API key](https://developer.twitter.com/en/docs/basics/authentication/guides/access-tokens.html). You need this because technically you are downloading the data on your own, I only labeled the tweets I don't own them.

### Why can't I just download theh data directly? Why do I need these scripts, what are those ids? What is happening? ###

Twitter has a strong policy for public distribuition of user data. It is very important for users to feel safe on the Internet and we agree with that, but it's undeniable how machine learning algorithms can be usefull for new technologies to appear. We only host here the ids of the annotated tweets and a tool for reaching them. In order to download the material, you need to create an account on Twitter and generate a valid consumers key, then you cna use our script to search for the content (using the ids) and use the sentiment label attatched to it.

### What else is in this repository? ###

* The semi-supervised approach (CasSul) used in the work of BRUM, H. B. (2018) for extending a sentiment corpus.
 * The dissertation is available in: [Teses USP](http://www.teses.usp.br/teses/disponiveis/55/55134/tde-26102018-123022/pt-br.php)
 * The paper explaining CasSul is available in: [BRUM, H. B. and NUNES, M. G. V. (2018)](https://www.springer.com/us/book/9783319997216).
    - Published in the 13th edition of the International Conference on the Computational Processing of Portuguese (PROPOR 2018).
* Further results obtained in BRUM, H. B. (2018) not added to the final dissertation.

### I can not download the tweets, or the running codes do not work, where do I go? ###

* You can send an e-mail for henrico.brum@gmail.com and I will be happy to answer any questions and help anyway I can. :D