# Counterfactuals and causability in explainable artificial intelligence: Theory, algorithms, and applications

<img width="737" alt="Screen Shot 2023-03-15 at 5 42 22 pm" src="https://user-images.githubusercontent.com/48231558/225239961-64cd8b42-5bba-43cc-9785-b90bf522ea49.png">

This project involves a review of the literature on counterfactuals and causality in Explainable Artificial Intelligence (XAI). The review was conducted using several text mining techniques, including topic modelling analysis with the Latent Dirichlet Allocation (LDA) algorithm.

The paper resulting from this project has been published in Q1 Journal "Information Fusion".



### Research Questions
The following research questions were proposed to help identify knowledge gaps in the area of causality, causability, and counterfactuals in XAI:

RQ1: What are the main theoretical approaches for counterfactuals in XAI (Theory)?
RQ2: What are the main algorithms in XAI that use counterfactuals as a means to promote understandable causal explanations (Algorithms)?
RQ3: What are the sufficient and necessary conditions for a system to promote causability (Applications)?
RQ4: What are the pressing challenges and research opportunities in XAI systems that promote Causability?
Search Process
To address the research questions, three well-known Computer Science academic databases were used: Scopus, IEEE Xplore, and Web of Science (WoS). These databases were selected because they have good coverage of works on artificial intelligence and provide APIs to retrieve data with few restrictions. The following search query was used to retrieve academic papers related to explainability or interpretability and causality or counterfactuals:

(artificial AND intelligence) AND (xai OR explai* OR interpretab*) AND (caus* OR counterf*)

This search query was used to extract bibliometric information from different databases, such as publication titles, abstracts, keywords, and year. Initially, the search returned a total of IEEE Xplore (6878), Scopus (1116), and WoS (126) articles. Duplicate entries and results with missing entries were removed, reducing the search process to IEEE Xplore (4712), Scopus (709), and WoS (124). The search process is summarized in the PRISMA flow diagram shown in below.

<img width="542" alt="Screen Shot 2023-03-15 at 5 46 47 pm" src="https://user-images.githubusercontent.com/48231558/225241549-c0c5bf7b-3b80-458b-95bb-3f9d898d8b64.png">


To ensure that the initial search query retrieved publications that match the review's scope, a topic modeling analysis was conducted based on Latent Dirichlet Allocation (LDA) to refine the search results.




### Research Methodology
The research methodology involved the following steps:

Definition of the search criteria for relevant literature on counterfactuals and causality in XAI.
Execution of the search on various academic databases, including IEEE Xplore, Scopus, and WoS.
Topic modelling analysis to refine the search results using the LDA algorithm with inclusion and exclusion criteria.
Extraction of publication data (title, abstract, author keywords, and year), systematization, and analysis of the relevant literature on counterfactuals and causality in XAI.
Identification of biases and limitations in the review process.
Topic modelling is a natural language processing technique that involves uncovering the underlying semantic structure of a document collection based on a hierarchical Bayesian analysis. LDA is a topic model that classifies text in a document into a particular topic. In this project, LDA enabled the clustering of words in publications with a high likelihood of term co-occurrence and allowed for the interpretation of the topics in each cluster.

In this project, text mining techniques were applied to the title, abstract, and author keywords retrieved from the search query. Stop word removal, word tokenization, stemming, and lemmatization were applied to the text data. LDA was then used to analyze the term co-occurrences in each database. The best-performing LDA model contained a total of 4 topics.


LDA model output
Fig. 4: LDA model output. Inter-topic distance showing the marginal topic distributions (left) and the top 10 most relevant terms for each topic.
<img width="739" alt="Screen Shot 2023-03-15 at 5 41 52 pm" src="https://user-images.githubusercontent.com/48231558/225239877-0aa60429-371a-494a-bc25-2f66709c9285.png">

The LDA model's output is illustrated in Fig.4 in the article, with the inter-topic distance showing the marginal topic distributions (left) and the top 10 most relevant terms for each topic. Analysing Fig. 4, Topic 1 contained all the words that are of interest to the research questions proposed in this survey paper: explainability, causality, and artificial intelligence. Topic 2 captured words primarily related to data management and technology, while Topic 3 contained words related to the human aspect of explainable AI, such as cognition, mental, and human. Finally, Topic 4 contained words associated with XAI in healthcare. For this survey paper, publications classified as either Topic 1 or Topic 3 were selected.

The search results were reduced to IEEE Xplore (50), Scopus (85), and WoS (30) after manually selecting articles about "causability," "causal," and "counterfactual." These publications were then analyzed for the final set of documents.

Link to Paper
[Click here to read the paper](https://reader.elsevier.com/reader/sd/pii/S1566253521002281?token=865006708BB405F4449B3B64D5A425039A9E0EE26E3B15CB4E163013A182FD0769C162540B9AD361944CB3
Acknowledgments
This project was published in Q1 Journal "Information Fusion".
