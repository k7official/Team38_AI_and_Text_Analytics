**Task 3: Comparative Corpus Analysis**  
Analysing changes in scientific abstracts over time can provide data-driven insights into the changing priorities and methods of researchers. These abstracts communicate how researchers frame problems, and the kind of claims that they highlight, as well as trends in applying specific methods.

**Objective:** Given a corpus of scientific abstracts, analyse how the focus, framing and communication of research has changed within the fields of AI, machine learning and NLP (you can either treat these as a single field or pick one to focus on). Analyse how different text analytics methods affect the insights that can be drawn.  
(Consider the field- AI as overall for the coursework)

The dataset is available at the HuggingFace page and contains metadata such as title, author, and journal. The webpage provides further documentation, and the Text Analytics Github repository provides a script for loading a suitable corpus for your project.  
Student group work:

1. Design and implement a text analysis pipeline to answer the above question:  
   1. Define the key steps in the pipeline, e.g., splitting corpus into periods, preprocessing, text representation (TF-IDF vectors, embeddings, etc.), labelling or describing each period, comparing periods.  
   2. Choose two important analytic axes to explore, such as: text representation (e.g., words, phrases, topics, embeddings); comparison method (keyword analysis, distance measures, classifying text into periods); temporal granularity; preprocessing choices.  
   3. For each design axis, compare 2-3 alternative approaches, motivated by a clear hypothesis about how each approach will perform.  
2. Examine the dates where temporal drift occurs by plotting abstracts in a dimensionally reduced space. Apply PCA to TF-IDF or embedding space. Project abstracts over time and visualise the trajectory of the field in the low-dimensional space.  
3. Present your main findings from the corpus in answer to the comparison question.  
4. Evaluate each tested approach in your pipeline in terms of:  
   1. Consistency and robustness of findings across methods  
   2. Interpretability of output.  
   3. Alignment with known or plausible patterns.  
5. Discuss trade-offs between approaches and the limitations of your approach.