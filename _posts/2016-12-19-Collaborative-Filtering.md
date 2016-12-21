---
layout: post
title: A concise tutorial on recommender systems
---

A recommender system predicts the likelihood that a user would prefer an item. Based on previous user interaction with the data source that the systems takes the information from (besides the data from other users, or historical trends), the system is capable to recommend an item to a user. Think about the fact that Amazon recommends you books that they think you could like; Amazon might be making effective use of a recommender system behind the curtains. This simple definition, allows us to think in a diverse set of applications where recommender systems might be useful. Applications such as documents, movies, music, romantic partners, or who to follow on Twitter, are pervasive and widely known in the world of Information Retrieval.

![_config.yml]({{ site.baseurl }}/images/posts/2016-12-19-Collaborative-Filtering/amazon_recommendation.png)

Such amazing applications, carries a huge amount of theory behind them. While theory can be a little bit intimidating and dry, basic understanding of data structures, a programming language, and a little bit of abstraction is all you need to understand the basics of recommender systems. 

In this tutorial, We will help you gain a basic understanding on collaborative based Recommender Systems, by building the most basic recommender system out there. We hope that this tutorial motivates you to find out more abour Recommender Sysyems, both in theory and practice. The prerequisites to reading this tutorial are knowledge of a programming language (we'll use Python, but if you know how does Hash Maps and List works, that's ok), and a little bit of high-school algebra. You do not need to have prior exposure to recommender systems.

This tutorial makes use of a class of RS (Recommender System) algorithm called **collaborative filtering**. A collaborative filtering algorithm works by finding a set of people (assuming persons are the only client or user of a RS) with preferences or tastes similar to the target user. Using this smlaller set of "similar" people, it constructs a ranked list of suggestions. There're are several ways to measure the similarity of two people. It's important to highlight that we're not going to use attriutes or descriptors of an item to recommend it, we're just using the tastes or preferences over that item.

Assuming that our users are people, and our items are simply that: items, we need to organize our data to ease the processing step. We're assuming that the data fits in memory, and that you can organize the data as follows.

![_config.yml]({{ site.baseurl }}/images/posts/2016-12-19-Collaborative-Filtering/image_collfilt.png)

The data structure that we are going to use, consists of people pointing to a dictionary whose keys are the items, and values are the numeric preference of each person on this item. If a person has never ranked the item, `C[i, j]`, is `null`. In this notation `C[i, j]` represents the numeric rating of `Person j`, over the `Item i`. No matter how the rating is expressed, we need to convert them to numeric values. A sample data structure for our working example is the following definition of a Python dictionary, it includes the rating of some remarkable people to some computer science topics.

```python
data = {
	'Alan Perlis': { 
		'Artificial intelligence': 1.46, 
		'Systems programming': 5.0, 
		'Software engineering': 3.34, 
		'Databases': 2.32
	},

	'Marvin Minsky': { 
		'Artificial intelligence': 5.0, 
		'Systems programming': 2.54,
		'Computation': 4.32, 
		'Algorithms': 2.76
	},

	'John McCarthy': { 
		'Artificial intelligence': 5.0, 
		'Programming language theory': 4.72, 
		'Systems programming': 3.25, 
		'Concurrency': 3.61, 
		'Formal methods': 3.58,
		'Computation': 3.23, 
		'Algorithms': 3.03 
	},

	'Edsger Dijkstra': { 
		'Programming language theory': 4.34, 
		'Systems programming': 4.52,
		'Software engineering': 4.04, 
		'Concurrency': 3.97,
		'Formal methods': 5.0, 
		'Algorithms': 4.92 
	},

	'Donald Knuth': { 
		'Programming language theory': 4.33, 
		'Systems programming': 3.57,
		'Computation': 4.39, 
		'Algorithms': 5.0 
	},

	'John Backus': { 
		'Programming language theory': 4.58, 
		'Systems programming': 4.43,
		'Software engineering': 4.38, 
		'Formal methods': 2.42, 
		'Databases': 2.80 
	},

	'Robert Floyd': { 
		'Programming language theory': 4.24, 
		'Systems programming': 2.17,
		'Concurrency': 2.92, 
		'Formal methods': 5.0, 
		'Computation': 3.18, 
		'Algorithms': 5.0 
	},

	'Tony Hoare': { 
		'Programming language theory': 4.64, 
		'Systems programming': 4.38,
		'Software engineering': 3.62, 
		'Concurrency': 4.88,
		'Formal methods': 4.72, 
		'Algorithms': 4.38
	},

	'Edgar Codd': { 
		'Systems programming': 4.60, 
		'Software engineering': 3.54,
		'Concurrency': 4.28, 
		'Formal methods': 1.53, 
		'Databases': 5.0
	},

	'Dennis Ritchie': { 
		'Programming language theory': 3.45, 
		'Systems programming': 5.0,
		'Software engineering': 4.83,
	},

	'Niklaus Wirth': { 
		'Programming language theory': 4.23, 
		'Systems programming': 4.22,
		'Software engineering': 4.74, 
		'Formal methods': 3.83, 
		'Algorithms': 3.95
	},

	'Robin Milner': { 
		'Programming language theory': 5.0, 
		'Systems programming': 1.66,
		'Concurrency': 4.62, 
		'Formal methods': 3.94,
	},

	'Leslie Lamport': { 
		'Programming language theory': 1,5, 
		'Systems programming': 2.76,
		'Software engineering': 3.76, 
		'Concurrency': 5.0,
		'Formal methods': 4.93, 
		'Algorithms': 4.63
	},

	'Michael Stonebraker': { 
		'Systems programming': 4.67, 
		'Software engineering': 3.86,
		'Concurrency': 4.14, 
		'Databases': 5.0,
	},
}
```

In this example, `Leslie Lamport`, rates `Software engineering` with `3.76`, while `Robin Milner`, rates `Programming language theory` with `5.0`. A simple problem that we might want to solve using this dataset and a recommender system, is how likely is `Marvin Minsky` to like `Programming language theory`.

