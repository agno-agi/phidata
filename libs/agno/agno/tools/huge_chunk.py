from agno.tools import Toolkit


class HugeTextChunkReturnTool(Toolkit):
    def __init__(self):
        super().__init__(name="huge_text_chunk_return_tool")
        self.register(self.get_huge_text_chunk)

    def get_huge_text_chunk(self) -> str:
        """
        Returns a huge text chunk.
        """
        return """
Perhaps you would like to give your homemade robot a brain of its own? Make it rec‐
ognize faces? Or learn to walk around?
Or maybe your company has tons of data (user logs, financial data, production data,
machine sensor data, hotline stats, HR reports, etc.), and more than likely you could
unearth some hidden gems if you just knew where to look. With Machine Learning,
you could accomplish the following and more:
• Segment customers and find the best marketing strategy for each group.
• Recommend products for each client based on what similar clients bought.
• Detect which transactions are likely to be fraudulent.
• Forecast next year’s revenue.
Whatever the reason, you have decided to learn Machine Learning and implement it
in your projects. Great idea!
Objective and Approach
This book assumes that you know close to nothing about Machine Learning. Its goal
is to give you the concepts, tools, and intuition you need to implement programs
capable of learning from data.
We will cover a large number of techniques, from the simplest and most commonly
used (such as Linear Regression) to some of the Deep Learning techniques that regu‐
larly win competitions.
Rather than implementing our own toy versions of each algorithm, we will be using
production-ready Python frameworks:
• Scikit-Learn is very easy to use, yet it implements many Machine Learning algo‐
rithms efficiently, so it makes for a great entry point to learning Machine Learn‐
ing. It was created by David Cournapeau in 2007, and is now led by a team of
researchers at the French Institute for Research in Computer Science and Auto‐
mation (Inria).
• TensorFlow is a more complex library for distributed numerical computation. It
makes it possible to train and run very large neural networks efficiently by dis‐
tributing the computations across potentially hundreds of multi-GPU (graphics
processing unit) servers. TensorFlow (TF) was created at Google and supports
many of its large-scale Machine Learning applications. It was open sourced in
November 2015, and version 2.0 was released in September 2019.
• Keras is a high-level Deep Learning API that makes it very simple to train and
run neural networks. It can run on top of either TensorFlow, Theano, or Micro‐
soft Cognitive Toolkit (formerly known as CNTK). TensorFlow comes with its
own implementation of this API, called tf.keras, which provides support for some
advanced TensorFlow features (e.g., the ability to efficiently load data).
The book favors a hands-on approach, growing an intuitive understanding of
Machine Learning through concrete working examples and just a little bit of theory.
While you can read this book without picking up your laptop, I highly recommend
you experiment with the code examples available online as Jupyter notebooks at
https://github.com/ageron/handson-ml2.
Prerequisites
This book assumes that you have some Python programming experience and that you
are familiar with Python’s main scientific libraries—in particular, NumPy, pandas,
and Matplotlib.
Also, if you care about what’s under the hood, you should have a reasonable under‐
standing of college-level math as well (calculus, linear algebra, probabilities, and sta‐
tistics).
If you don’t know Python yet, http://learnpython.org/ is a great place to start. The offi‐
cial tutorial on Python.org is also quite good.
If you have never used Jupyter, Chapter 2 will guide you through installation and the
basics: it is a powerful tool to have in your toolbox.
If you are not familiar with Python’s scientific libraries, the provided Jupyter note‐
books include a few tutorials. There is also a quick math tutorial for linear algebra.
Roadmap
This book is organized in two parts. Part I, The Fundamentals of Machine Learning,
covers the following topics:
• What Machine Learning is, what problems it tries to solve, and the main cate‐
gories and fundamental concepts of its systems
• The steps in a typical Machine Learning project
• Learning by fitting a model to data
• Optimizing a cost function
• Handling, cleaning, and preparing data
• Selecting and engineering features
• Selecting a model and tuning hyperparameters using cross-validation
• The challenges of Machine Learning, in particular underfitting and overfitting
(the bias/variance trade-off)
• The most common learning algorithms: Linear and Polynomial Regression,
Logistic Regression, k-Nearest Neighbors, Support Vector Machines, Decision
Trees, Random Forests, and Ensemble methods
• Reducing the dimensionality of the training data to fight the “curse of dimen‐
sionality”
• Other unsupervised learning techniques, including clustering, density estima‐
tion, and anomaly detection
Part II, Neural Networks and Deep Learning, covers the following topics:
• What neural nets are and what they’re good for
• Building and training neural nets using TensorFlow and Keras
• The most important neural net architectures: feedforward neural nets for tabular
data, convolutional nets for computer vision, recurrent nets and long short-term
memory (LSTM) nets for sequence processing, encoder/decoders and Trans‐
formers for natural language processing, autoencoders and generative adversarial
networks (GANs) for generative learning
• Techniques for training deep neural nets
• How to build an agent (e.g., a bot in a game) that can learn good strategies
through trial and error, using Reinforcement Learning
• Loading and preprocessing large amounts of data efficiently
• Training and deploying TensorFlow models at scale
The first part is based mostly on Scikit-Learn, while the second part uses TensorFlow
and Keras.
Don’t jump into deep waters too hastily: while Deep Learning is no
doubt one of the most exciting areas in Machine Learning, you
should master the fundamentals first. Moreover, most problems
can be solved quite well using simpler techniques such as Random
Forests and Ensemble methods (discussed in Part I). Deep Learn‐
ing is best suited for complex problems such as image recognition,
speech recognition, or natural language processing, provided you
have enough data, computing power, and patience.
Changes in the Second Edition
This second edition has six main objectives:
1. Cover additional ML topics: more unsupervised learning techniques (including
clustering, anomaly detection, density estimation, and mixture models); more
techniques for training deep nets (including self-normalized networks); addi‐
tional computer vision techniques (including Xception, SENet, object detection
with YOLO, and semantic segmentation using R-CNN); handling sequences
using covolutional neural networks (CNNs, including WaveNet); natural lan‐
guage processing using recurrent neural networks (RNNs), CNNs, and Trans‐
formers; and GANs.
2. Cover additional libraries and APIs (Keras, the Data API, TF-Agents for Rein‐
forcement Learning) and training and deploying TF models at scale using the
Distribution Strategies API, TF-Serving, and Google Cloud AI Platform. Also
briefly introduce TF Transform, TFLite, TF Addons/Seq2Seq, and TensorFlow.js.
3. Discuss some of the latest important results from Deep Learning research.
4. Migrate all TensorFlow chapters to TensorFlow 2, and use TensorFlow’s imple‐
mentation of the Keras API (tf.keras) whenever possible.
5. Update the code examples to use the latest versions of Scikit-Learn, NumPy, pan‐
das, Matplotlib, and other libraries.
6. Clarify some sections and fix some errors, thanks to plenty of great feedback
from readers.
Some chapters were added, others were rewritten, and a few were reordered. See
https://homl.info/changes2 for more details on what changed in the second edition.
Other Resources
Many excellent resources are available to learn about Machine Learning. For example,
Andrew Ng’s ML course on Coursera is amazing, although it requires a significant
time investment (think months).
There are also many interesting websites about Machine Learning, including of
course Scikit-Learn’s exceptional User Guide. You may also enjoy Dataquest, which
provides very nice interactive tutorials, and ML blogs such as those listed on Quora.
Finally, the Deep Learning website has a good list of resources to check out to learn
more.
There are many other introductory books about Machine Learning. In particular:
Preface | xix
• Joel Grus’s Data Science from Scratch (O’Reilly) presents the fundamentals of
Machine Learning and implements some of the main algorithms in pure Python
(from scratch, as the name suggests).
• Stephen Marsland’s Machine Learning: An Algorithmic Perspective (Chapman &
Hall) is a great introduction to Machine Learning, covering a wide range of topics
in depth with code examples in Python (also from scratch, but using NumPy).
• Sebastian Raschka’s Python Machine Learning (Packt Publishing) is also a great
introduction to Machine Learning and leverages Python open source libraries
(Pylearn 2 and Theano).
• François Chollet’s Deep Learning with Python (Manning) is a very practical book
that covers a large range of topics in a clear and concise way, as you might expect
from the author of the excellent Keras library. It favors code examples over math‐
ematical theory.
• Andriy Burkov’s The Hundred-Page Machine Learning Book is very short and cov‐
ers an impressive range of topics, introducing them in approachable terms
without shying away from the math equations.
• Yaser S. Abu-Mostafa, Malik Magdon-Ismail, and Hsuan-Tien Lin’s Learning from
Data (AMLBook) is a rather theoretical approach to ML that provides deep
insights, in particular on the bias/variance trade-off (see Chapter 4).
• Stuart Russell and Peter Norvig’s Artificial Intelligence: A Modern Approach, 3rd
Edition (Pearson), is a great (and huge) book covering an incredible amount of
topics, including Machine Learning. It helps put ML into perspective.
Finally, joining ML competition websites such as Kaggle.com will allow you to prac‐
tice your skills on real-world problems, with help and insights from some of the best
ML professionals out there.
Conventions Used in This Book
The following typographical conventions are used in this book:
Italic
Indicates new terms, URLs, email addresses, filenames, and file extensions.
Constant width
Used for program listings, as well as within paragraphs to refer to program ele‐
ments such as variable or function names, databases, data types, environment
variables, statements and keywords.
Constant width bold
Shows commands or other text that should be typed literally by the user.
xx | Preface
Constant width italic
Shows text that should be replaced with user-supplied values or by values deter‐
mined by context.
This element signifies a tip or suggestion.
This element signifies a general note.
This element indicates a warning or caution.
Code Examples
There is a series of Jupyter notebooks full of supplemental material, such as code
examples and exercises, available for download at https://github.com/ageron/handson-
ml2.
Some of the code examples in the book leave out repetitive sections or details that are
obvious or unrelated to Machine Learning. This keeps the focus on the important
parts of the code and saves space to cover more topics. If you want the full code
examples, they are all available in the Jupyter notebooks.
Note that when the code examples display some outputs, these code examples are
shown with Python prompts (>>> and ...), as in a Python shell, to clearly distinguish
the code from the outputs. For example, this code defines the square() function,
then it computes and displays the square of 3:
>>> def square(x):
... return x ** 2
...
>>> result = square(3)
>>> result
9
When code does not display anything, prompts are not used. However, the result may
sometimes be shown as a comment, like this:
Preface | xxi
def square(x):
return x ** 2
result = square(3) # result is 9
Using Code Examples
This book is here to help you get your job done. In general, if example code is offered
with this book, you may use it in your programs and documentation. You do not
need to contact us for permission unless you’re reproducing a significant portion of
the code. For example, writing a program that uses several chunks of code from this
book does not require permission. Selling or distributing a CD-ROM of examples
from O’Reilly books does require permission. Answering a question by citing this
book and quoting example code does not require permission. Incorporating a signifi‐
cant amount of example code from this book into your product’s documentation does
require permission.
We appreciate, but do not require, attribution. An attribution usually includes the
title, author, publisher, and ISBN. For example: “Hands-On Machine Learning with
Scikit-Learn, Keras, and TensorFlow, 2nd Edition, by Aurélien Géron (O’Reilly).
Copyright 2019 Kiwisoft S.A.S., 978-1-492-03264-9.” If you feel your use of code
examples falls outside fair use or the permission given above, feel free to contact us at
permissions@oreilly.com.
O’Reilly Online Learning
For almost 40 years, O’Reilly Media has provided technology
and business training, knowledge, and insight to help compa‐
nies succeed.
Our unique network of experts and innovators share their knowledge and expertise
through books, articles, conferences, and our online learning platform. O’Reilly’s
online learning platform gives you on-demand access to live training courses, in-
depth learning paths, interactive coding environments, and a vast collection of text
and video from O’Reilly and 200+ other publishers. For more information, please
visit http://oreilly.com.
xxii | Preface
How to Contact Us
Please address comments and questions concerning this book to the publisher:
O’Reilly Media, Inc.
1005 Gravenstein Highway North
Sebastopol, CA 95472
800-998-9938 (in the United States or Canada)
707-829-0515 (international or local)
707-829-0104 (fax)
We have a web page for this book, where we list errata, examples, and any additional
information. You can access this page at https://homl.info/oreilly2.
To comment or ask technical questions about this book, send email to bookques‐
tions@oreilly.com.
For more information about our books, courses, conferences, and news, see our web‐
site at http://www.oreilly.com.
Find us on Facebook: http://facebook.com/oreilly
Follow us on Twitter: http://twitter.com/oreillymedia
Watch us on YouTube: http://www.youtube.com/oreillymedia
Acknowledgments
Never in my wildest dreams did I imagine that the first edition of this book would get
such a large audience. I received so many messages from readers, many asking ques‐
tions, some kindly pointing out errata, and most sending me encouraging words. I
cannot express how grateful I am to all these readers for their tremendous support.
Thank you all so very much! Please do not hesitate to file issues on GitHub if you find
errors in the code examples (or just to ask questions), or to submit errata if you find
errors in the text. Some readers also shared how this book helped them get their first
job, or how it helped them solve a concrete problem they were working on. I find
such feedback incredibly motivating. If you find this book helpful, I would love it if
you could share your story with me, either privately (e.g., via LinkedIn) or publicly
(e.g., in a tweet or through an Amazon review).
I am also incredibly thankful to all the amazing people who took time out of their
busy lives to review my book with such care. In particular, I would like to thank Fran‐
çois Chollet for reviewing all the chapters based on Keras and TensorFlow and giving
me some great in-depth feedback. Since Keras is one of the main additions to this sec‐
ond edition, having its author review the book was invaluable. I highly recommend
Preface | xxiii
François’s book Deep Learning with Python (Manning): it has the conciseness, clarity,
and depth of the Keras library itself. Special thanks as well to Ankur Patel, who
reviewed every chapter of this second edition and gave me excellent feedback, in par‐
ticular on Chapter 9, which covers unsupervised learning techniques. He could write
a whole book on the topic… oh, wait, he did! Do check out Hands-On Unsupervised
Learning Using Python: How to Build Applied Machine Learning Solutions from Unla‐
beled Data (O’Reilly). Huge thanks as well to Olzhas Akpambetov, who reviewed all
the chapters in the second part of the book, tested much of the code, and offered
many great suggestions. I’m grateful to Mark Daoust, Jon Krohn, Dominic Monn,
and Josh Patterson for reviewing the second part of this book so thoroughly and
offering their expertise. They left no stone unturned and provided amazingly useful
feedback.
While writing this second edition, I was fortunate enough to get plenty of help from
members of the TensorFlow team—in particular Martin Wicke, who tirelessly
answered dozens of my questions and dispatched the rest to the right people, includ‐
ing Karmel Allison, Paige Bailey, Eugene Brevdo, William Chargin, Daniel “Wolff ”
Dobson, Nick Felt, Bruce Fontaine, Goldie Gadde, Sandeep Gupta, Priya Gupta,
Kevin Haas, Konstantinos Katsiapis ,Viacheslav Kovalevskyi, Allen Lavoie, Clemens
Mewald, Dan Moldovan, Sean Morgan, Tom O’Malley, Alexandre Passos, André Sus‐
ano Pinto, Anthony Platanios, Oscar Ramirez, Anna Revinskaya, Saurabh Saxena,
Ryan Sepassi, Jiri Simsa, Xiaodan Song, Christina Sorokin, Dustin Tran, Todd Wang,
Pete Warden (who also reviewed the first edition) Edd Wilder-James, and Yuefeng
Zhou, all of whom were tremendously helpful. Huge thanks to all of you, and to all
other members of the TensorFlow team, not just for your help, but also for making
such a great library! Special thanks to Irene Giannoumis and Robert Crowe of the
TFX team for reviewing Chapters 13 and 19 in depth.
Many thanks as well to O’Reilly’s fantastic staff, in particular Nicole Taché, who gave
me insightful feedback and was always cheerful, encouraging, and helpful: I could not
dream of a better editor. Big thanks to Michele Cronin as well, who was very helpful
(and patient) at the start of this second edition, and to Kristen Brown, the production
editor for the second edition, who saw it through all the steps (she also coordinated
fixes and updates for each reprint of the first edition). Thanks as well to Rachel Mon‐
aghan and Amanda Kersey for their thorough copyediting (respectively for the first
and second edition), and to Johnny O’Toole who managed the relationship with
Amazon and answered many of my questions. Thanks to Marie Beaugureau, Ben
Lorica, Mike Loukides, and Laurel Ruma for believing in this project and helping me
define its scope. Thanks to Matt Hacker and all of the Atlas team for answering all my
technical questions regarding formatting, AsciiDoc, and LaTeX, and thanks to Nick
Adams, Rebecca Demarest, Rachel Head, Judith McConville, Helen Monroe, Karen
Montgomery, Rachel Roumeliotis, and everyone else at O’Reilly who contributed to
this book.
xxiv | Preface
I would also like to thank my former Google colleagues, in particular the YouTube
video classification team, for teaching me so much about Machine Learning. I could
never have started the first edition without them. Special thanks to my personal ML
gurus: Clément Courbet, Julien Dubois, Mathias Kende, Daniel Kitachewsky, James
Pack, Alexander Pak, Anosh Raj, Vitor Sessak, Wiktor Tomczak, Ingrid von Glehn,
and Rich Washington. And thanks to everyone else I worked with at YouTube and in
the amazing Google research teams in Mountain View. Many thanks as well to Martin
Andrews, Sam Witteveen, and Jason Zaman for welcoming me into their Google
Developer Experts group in Singapore, with the kind support of Soonson Kwon, and
for all the great discussions we had about Deep Learning and TensorFlow. Anyone
interested in Deep Learning in Singapore should definitely join their Deep Learning
Singapore meetup. Jason deserves special thanks for sharing some of his TFLite
expertise for Chapter 19!
I will never forget the kind people who reviewed the first edition of this book, includ‐
ing David Andrzejewski, Lukas Biewald, Justin Francis, Vincent Guilbeau, Eddy
Hung, Karim Matrah, Grégoire Mesnil, Salim Sémaoune, Iain Smears, Michel Tessier,
Ingrid von Glehn, Pete Warden, and of course my dear brother Sylvain. Special
thanks to Haesun Park, who gave me plenty of excellent feedback and caught several
errors while he was writing the Korean translation of the first edition of this book. He
also translated the Jupyter notebooks into Korean, not to mention TensorFlow’s doc‐
umentation. I do not speak Korean, but judging by the quality of his feedback, all his
translations must be truly excellent! Haesun also kindly contributed some of the solu‐
tions to the exercises in this second edition.
Last but not least, I am infinitely grateful to my beloved wife, Emmanuelle, and to our
three wonderful children, Alexandre, Rémi, and Gabrielle, for encouraging me to
work hard on this book. I’m also thankful to them for their insatiable curiosity:
explaining some of the most difficult concepts in this book to my wife and children
helped me clarify my thoughts and directly improved many parts of it. And they keep
bringing me cookies and coffee! What more can one dream of?
Preface | xxv
PART I
The Fundamentals of
Machine Learning
CHAPTER 1
The Machine Learning Landscape
When most people hear “Machine Learning,” they picture a robot: a dependable but‐
ler or a deadly Terminator, depending on who you ask. But Machine Learning is not
just a futuristic fantasy; it’s already here. In fact, it has been around for decades in
some specialized applications, such as Optical Character Recognition (OCR). But the
first ML application that really became mainstream, improving the lives of hundreds
of millions of people, took over the world back in the 1990s: the spam filter. It’s not
exactly a self-aware Skynet, but it does technically qualify as Machine Learning (it has
actually learned so well that you seldom need to flag an email as spam anymore). It
was followed by hundreds of ML applications that now quietly power hundreds of
products and features that you use regularly, from better recommendations to voice
search.
Where does Machine Learning start and where does it end? What exactly does it
mean for a machine to learn something? If I download a copy of Wikipedia, has my
computer really learned something? Is it suddenly smarter? In this chapter we will
start by clarifying what Machine Learning is and why you may want to use it.
Then, before we set out to explore the Machine Learning continent, we will take a
look at the map and learn about the main regions and the most notable landmarks:
supervised versus unsupervised learning, online versus batch learning, instance-
based versus model-based learning. Then we will look at the workflow of a typical ML
project, discuss the main challenges you may face, and cover how to evaluate and
fine-tune a Machine Learning system.
This chapter introduces a lot of fundamental concepts (and jargon) that every data
scientist should know by heart. It will be a high-level overview (it’s the only chapter
without much code), all rather simple, but you should make sure everything is crystal
clear to you before continuing on to the rest of the book. So grab a coffee and let’s get
started!
1
If you already know all the Machine Learning basics, you may want
to skip directly to Chapter 2. If you are not sure, try to answer all
the questions listed at the end of the chapter before moving on.
What Is Machine Learning?
Machine Learning is the science (and art) of programming computers so they can
learn from data.
Here is a slightly more general definition:
[Machine Learning is the] field of study that gives computers the ability to learn
without being explicitly programmed.
—Arthur Samuel, 1959
And a more engineering-oriented one:
A computer program is said to learn from experience E with respect to some task T
and some performance measure P, if its performance on T, as measured by P,
improves with experience E.
—Tom Mitchell, 1997
Your spam filter is a Machine Learning program that, given examples of spam emails
(e.g., flagged by users) and examples of regular (nonspam, also called “ham”) emails,
can learn to flag spam. The examples that the system uses to learn are called the train‐
ing set. Each training example is called a training instance (or sample). In this case, the
task T is to flag spam for new emails, the experience E is the training data, and the
performance measure P needs to be defined; for example, you can use the ratio of
correctly classified emails. This particular performance measure is called accuracy,
and it is often used in classification tasks.
If you just download a copy of Wikipedia, your computer has a lot more data, but it is
not suddenly better at any task. Thus, downloading a copy of Wikipedia is not
Machine Learning.
Why Use Machine Learning?
Consider how you would write a spam filter using traditional programming techni‐
ques (Figure 1-1):
1. First you would consider what spam typically looks like. You might notice that
some words or phrases (such as “4U,” “credit card,” “free,” and “amazing”) tend to
come up a lot in the subject line. Perhaps you would also notice a few other pat‐
terns in the sender’s name, the email’s body, and other parts of the email.
2 | Chapter 1: The Machine Learning Landscape
2. You would write a detection algorithm for each of the patterns that you noticed,
and your program would flag emails as spam if a number of these patterns were
detected.
3. You would test your program and repeat steps 1 and 2 until it was good enough
to launch.
Figure 1-1. The traditional approach
Since the problem is difficult, your program will likely become a long list of complex
rules—pretty hard to maintain.
In contrast, a spam filter based on Machine Learning techniques automatically learns
which words and phrases are good predictors of spam by detecting unusually fre‐
quent patterns of words in the spam examples compared to the ham examples
(Figure 1-2). The program is much shorter, easier to maintain, and most likely more
accurate.
What if spammers notice that all their emails containing “4U” are blocked? They
might start writing “For U” instead. A spam filter using traditional programming
techniques would need to be updated to flag “For U” emails. If spammers keep work‐
ing around your spam filter, you will need to keep writing new rules forever.
In contrast, a spam filter based on Machine Learning techniques automatically noti‐
ces that “For U” has become unusually frequent in spam flagged by users, and it starts
flagging them without your intervention (Figure 1-3).
Why Use Machine Learning? | 3
Figure 1-2. The Machine Learning approach
Figure 1-3. Automatically adapting to change
Another area where Machine Learning shines is for problems that either are too com‐
plex for traditional approaches or have no known algorithm. For example, consider
speech recognition. Say you want to start simple and write a program capable of dis‐
tinguishing the words “one” and “two.” You might notice that the word “two” starts
with a high-pitch sound (“T”), so you could hardcode an algorithm that measures
high-pitch sound intensity and use that to distinguish ones and twos—but obviously
this technique will not scale to thousands of words spoken by millions of very differ‐
ent people in noisy environments and in dozens of languages. The best solution (at
least today) is to write an algorithm that learns by itself, given many example record‐
ings for each word.
Finally, Machine Learning can help humans learn (Figure 1-4). ML algorithms can be
inspected to see what they have learned (although for some algorithms this can be
tricky). For instance, once a spam filter has been trained on enough spam, it can
easily be inspected to reveal the list of words and combinations of words that it
believes are the best predictors of spam. Sometimes this will reveal unsuspected
4 | Chapter 1: The Machine Learning Landscape
correlations or new trends, and thereby lead to a better understanding of the prob‐
lem. Applying ML techniques to dig into large amounts of data can help discover pat‐
terns that were not immediately apparent. This is called data mining.
Figure 1-4. Machine Learning can help humans learn
To summarize, Machine Learning is great for:
• Problems for which existing solutions require a lot of fine-tuning or long lists of
rules: one Machine Learning algorithm can often simplify code and perform bet‐
ter than the traditional approach.
• Complex problems for which using a traditional approach yields no good solu‐
tion: the best Machine Learning techniques can perhaps find a solution.
• Fluctuating environments: a Machine Learning system can adapt to new data.
• Getting insights about complex problems and large amounts of data.
Examples of Applications
Let’s look at some concrete examples of Machine Learning tasks, along with the tech‐
niques that can tackle them:
Analyzing images of products on a production line to automatically classify them
This is image classification, typically performed using convolutional neural net‐
works (CNNs; see Chapter 14).
Examples of Applications | 5
Detecting tumors in brain scans
This is semantic segmentation, where each pixel in the image is classified (as we
want to determine the exact location and shape of tumors), typically using CNNs
as well.
Automatically classifying news articles
This is natural language processing (NLP), and more specifically text classifica‐
tion, which can be tackled using recurrent neural networks (RNNs), CNNs, or
Transformers (see Chapter 16).
Automatically flagging offensive comments on discussion forums
This is also text classification, using the same NLP tools.
Summarizing long documents automatically
This is a branch of NLP called text summarization, again using the same tools.
Creating a chatbot or a personal assistant
This involves many NLP components, including natural language understanding
(NLU) and question-answering modules.
Forecasting your company’s revenue next year, based on many performance metrics
This is a regression task (i.e., predicting values) that may be tackled using any
regression model, such as a Linear Regression or Polynomial Regression model
(see Chapter 4), a regression SVM (see Chapter 5), a regression Random Forest
(see Chapter 7), or an artificial neural network (see Chapter 10). If you want to
take into account sequences of past performance metrics, you may want to use
RNNs, CNNs, or Transformers (see Chapters 15 and 16).
Making your app react to voice commands
This is speech recognition, which requires processing audio samples: since they
are long and complex sequences, they are typically processed using RNNs, CNNs,
or Transformers (see Chapters 15 and 16).
Detecting credit card fraud
This is anomaly detection (see Chapter 9).
Segmenting clients based on their purchases so that you can design a different marketing
strategy for each segment
This is clustering (see Chapter 9).
Representing a complex, high-dimensional dataset in a clear and insightful diagram
This is data visualization, often involving dimensionality reduction techniques
(see Chapter 8).
Recommending a product that a client may be interested in, based on past purchases
This is a recommender system. One approach is to feed past purchases (and
other information about the client) to an artificial neural network (see Chap‐
6 | Chapter 1: The Machine Learning Landscape
ter 10), and get it to output the most likely next purchase. This neural net would
typically be trained on past sequences of purchases across all clients.
Building an intelligent bot for a game
This is often tackled using Reinforcement Learning (RL; see Chapter 18), which
is a branch of Machine Learning that trains agents (such as bots) to pick the
actions that will maximize their rewards over time (e.g., a bot may get a reward
every time the player loses some life points), within a given environment (such as
the game). The famous AlphaGo program that beat the world champion at the
game of Go was built using RL.
This list could go on and on, but hopefully it gives you a sense of the incredible
breadth and complexity of the tasks that Machine Learning can tackle, and the types
of techniques that you would use for each task.
Types of Machine Learning Systems
There are so many different types of Machine Learning systems that it is useful to
classify them in broad categories, based on the following criteria:
• Whether or not they are trained with human supervision (supervised, unsuper‐
vised, semisupervised, and Reinforcement Learning)
• Whether or not they can learn incrementally on the fly (online versus batch
learning)
• Whether they work by simply comparing new data points to known data points,
or instead by detecting patterns in the training data and building a predictive
model, much like scientists do (instance-based versus model-based learning)
These criteria are not exclusive; you can combine them in any way you like. For
example, a state-of-the-art spam filter may learn on the fly using a deep neural net‐
work model trained using examples of spam and ham; this makes it an online, model-
based, supervised learning system.
Let’s look at each of these criteria a bit more closely.
Supervised/Unsupervised Learning
Machine Learning systems can be classified according to the amount and type of
supervision they get during training. There are four major categories: supervised
learning, unsupervised learning, semisupervised learning, and Reinforcement
Learning.
Types of Machine Learning Systems | 7
Supervised learning
In supervised learning, the training set you feed to the algorithm includes the desired
solutions, called labels (Figure 1-5).
Figure 1-5. A labeled training set for spam classification (an example of supervised
learning)
A typical supervised learning task is classification. The spam filter is a good example
of this: it is trained with many example emails along with their class (spam or ham),
and it must learn how to classify new emails.
Another typical task is to predict a target numeric value, such as the price of a car,
given a set of features (mileage, age, brand, etc.) called predictors. This sort of task is
called regression (Figure 1-6).1 To train the system, you need to give it many examples
of cars, including both their predictors and their labels (i.e., their prices).
In Machine Learning an attribute is a data type (e.g., “mileage”),
while a feature has several meanings, depending on the context, but
generally means an attribute plus its value (e.g., “mileage =
15,000”). Many people use the words attribute and feature inter‐
changeably.
Note that some regression algorithms can be used for classification as well, and vice
versa. For example, Logistic Regression is commonly used for classification, as it can
output a value that corresponds to the probability of belonging to a given class (e.g.,
20% chance of being spam).
1 Fun fact: this odd-sounding name is a statistics term introduced by Francis Galton while he was studying the
fact that the children of tall people tend to be shorter than their parents. Since the children were shorter, he
called this regression to the mean. This name was then applied to the methods he used to analyze correlations
between variables.
8 | Chapter 1: The Machine Learning Landscape
Figure 1-6. A regression problem: predict a value, given an input feature (there are usu‐
ally multiple input features, and sometimes multiple output values)
Here are some of the most important supervised learning algorithms (covered in this
book):
• k-Nearest Neighbors
• Linear Regression
• Logistic Regression
• Support Vector Machines (SVMs)
• Decision Trees and Random Forests
• Neural networks2
Unsupervised learning
In unsupervised learning, as you might guess, the training data is unlabeled
(Figure 1-7). The system tries to learn without a teacher.
2 Some neural network architectures can be unsupervised, such as autoencoders and restricted Boltzmann
machines. They can also be semisupervised, such as in deep belief networks and unsupervised pretraining.
Types of Machine Learning Systems | 9
Figure 1-7. An unlabeled training set for unsupervised learning
Here are some of the most important unsupervised learning algorithms (most of
these are covered in Chapters 8 and 9):
• Clustering
— K-Means
— DBSCAN
— Hierarchical Cluster Analysis (HCA)
• Anomaly detection and novelty detection
— One-class SVM
— Isolation Forest
• Visualization and dimensionality reduction
— Principal Component Analysis (PCA)
— Kernel PCA
— Locally Linear Embedding (LLE)
— t-Distributed Stochastic Neighbor Embedding (t-SNE)
• Association rule learning
— Apriori
— Eclat
For example, say you have a lot of data about your blog’s visitors. You may want to
run a clustering algorithm to try to detect groups of similar visitors (Figure 1-8). At
no point do you tell the algorithm which group a visitor belongs to: it finds those
connections without your help. For example, it might notice that 40% of your visitors
are males who love comic books and generally read your blog in the evening, while
20% are young sci-fi lovers who visit during the weekends. If you use a hierarchical
clustering algorithm, it may also subdivide each group into smaller groups. This may
help you target your posts for each group.
10 | Chapter 1: The Machine Learning Landscape
Figure 1-8. Clustering
Visualization algorithms are also good examples of unsupervised learning algorithms:
you feed them a lot of complex and unlabeled data, and they output a 2D or 3D rep‐
resentation of your data that can easily be plotted (Figure 1-9). These algorithms try
to preserve as much structure as they can (e.g., trying to keep separate clusters in the
input space from overlapping in the visualization) so that you can understand how
the data is organized and perhaps identify unsuspected patterns.
Figure 1-9. Example of a t-SNE visualization highlighting semantic clusters3
3 Notice how animals are rather well separated from vehicles and how horses are close to deer but far from
birds. Figure reproduced with permission from Richard Socher et al., “Zero-Shot Learning Through Cross-
Modal Transfer,” Proceedings of the 26th International Conference on Neural Information Processing Systems 1
(2013): 935–943.
Types of Machine Learning Systems | 11
A related task is dimensionality reduction, in which the goal is to simplify the data
without losing too much information. One way to do this is to merge several correla‐
ted features into one. For example, a car’s mileage may be strongly correlated with its
age, so the dimensionality reduction algorithm will merge them into one feature that
represents the car’s wear and tear. This is called feature extraction.
It is often a good idea to try to reduce the dimension of your train‐
ing data using a dimensionality reduction algorithm before you
feed it to another Machine Learning algorithm (such as a super‐
vised learning algorithm). It will run much faster, the data will take
up less disk and memory space, and in some cases it may also per‐
form better.
Yet another important unsupervised task is anomaly detection—for example, detect‐
ing unusual credit card transactions to prevent fraud, catching manufacturing defects,
or automatically removing outliers from a dataset before feeding it to another learn‐
ing algorithm. The system is shown mostly normal instances during training, so it
learns to recognize them; then, when it sees a new instance, it can tell whether it looks
like a normal one or whether it is likely an anomaly (see Figure 1-10). A very similar
task is novelty detection: it aims to detect new instances that look different from all
instances in the training set. This requires having a very “clean” training set, devoid of
any instance that you would like the algorithm to detect. For example, if you have
thousands of pictures of dogs, and 1% of these pictures represent Chihuahuas, then a
novelty detection algorithm should not treat new pictures of Chihuahuas as novelties.
On the other hand, anomaly detection algorithms may consider these dogs as so rare
and so different from other dogs that they would likely classify them as anomalies (no
offense to Chihuahuas).
Figure 1-10. Anomaly detection
Finally, another common unsupervised task is association rule learning, in which the
goal is to dig into large amounts of data and discover interesting relations between
12 | Chapter 1: The Machine Learning Landscape
attributes. For example, suppose you own a supermarket. Running an association rule
on your sales logs may reveal that people who purchase barbecue sauce and potato
chips also tend to buy steak. Thus, you may want to place these items close to one
another.
Semisupervised learning
Since labeling data is usually time-consuming and costly, you will often have plenty of
unlabeled instances, and few labeled instances. Some algorithms can deal with data
that’s partially labeled. This is called semisupervised learning (Figure 1-11).
Figure 1-11. Semisupervised learning with two classes (triangles and squares): the unla‐
beled examples (circles) help classify a new instance (the cross) into the triangle class
rather than the square class, even though it is closer to the labeled squares
Some photo-hosting services, such as Google Photos, are good examples of this. Once
you upload all your family photos to the service, it automatically recognizes that the
same person A shows up in photos 1, 5, and 11, while another person B shows up in
photos 2, 5, and 7. This is the unsupervised part of the algorithm (clustering). Now all
the system needs is for you to tell it who these people are. Just add one label per per‐
son4 and it is able to name everyone in every photo, which is useful for searching
photos.
Most semisupervised learning algorithms are combinations of unsupervised and
supervised algorithms. For example, deep belief networks (DBNs) are based on unsu‐
pervised components called restricted Boltzmann machines (RBMs) stacked on top of
one another. RBMs are trained sequentially in an unsupervised manner, and then the
whole system is fine-tuned using supervised learning techniques.
4 That’s when the system works perfectly. In practice it often creates a few clusters per person, and sometimes
mixes up two people who look alike, so you may need to provide a few labels per person and manually clean
up some clusters.
Types of Machine Learning Sys
"""
