package org.rag4j.jm;

import org.rag4j.indexing.ContentReader;
import org.rag4j.indexing.IndexingService;
import org.rag4j.indexing.Splitter;
import org.rag4j.indexing.splitters.SentenceSplitter;
import org.rag4j.integrations.ollama.OllamaAccess;
import org.rag4j.integrations.ollama.OllamaChatService;
import org.rag4j.integrations.ollama.OllamaEmbedder;
import org.rag4j.rag.embedding.Embedder;
import org.rag4j.rag.generation.AnswerGenerator;
import org.rag4j.rag.generation.ObservedAnswerGenerator;
import org.rag4j.rag.generation.chat.ChatService;
import org.rag4j.rag.generation.quality.AnswerQuality;
import org.rag4j.rag.generation.quality.AnswerQualityService;
import org.rag4j.rag.model.RelevantChunk;
import org.rag4j.rag.retrieval.ObservedRetriever;
import org.rag4j.rag.retrieval.RetrievalOutput;
import org.rag4j.rag.retrieval.RetrievalStrategy;
import org.rag4j.rag.retrieval.Retriever;
import org.rag4j.rag.retrieval.strategies.TopNRetrievalStrategy;
import org.rag4j.rag.store.local.InternalContentStore;
import org.rag4j.rag.tracker.LoggingRAGObserverPersistor;
import org.rag4j.rag.tracker.RAGObserver;
import org.rag4j.rag.tracker.RAGObserverPersistor;
import org.rag4j.rag.tracker.RAGTracker;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * <p>This is an application that demonstrates the use of the RAG library RAG4J. More information about the framework is
 * available at the <a href="https://rag4j.github.io/rag4j/">project's GitHub page</a>. A blog post is explaining the
 * steps to create this application. If you just want to run it, you came to the right place. You can make some changes
 * if you want to use other Splitters, strategies, or even other data sources.</p>
 * <p>There are two data files. You start with sessions-one.jsonl. This is faster to try different splitters. The
 * second dataset contains all the talks from the jFall conference 2024. Change the file in the constructor.</p>
 * <p>In this class, you use Ollama. Ollama runs different LLMs on you machine. You can use multiple models. Beware
 * that some models work better than others. THis project is tested with the <strong>llama3.2</strong> model. You can
 * configure other models. Check the <a href="https://ollama.com/search">Ollama website</a> for more information. The
 * constructor is the place to change the model if you want to.</p>
 * <p>For this application you use the {@link org.rag4j.rag.store.local.InternalContentStore}. This class uses a Map
 * to store the vectors and chunks. The store requires an embedder to create embeddings from the chunks you want to
 * store. You use the default Ollama model for the rag4j framework: <strong>nomic-embed-text</strong>.</p>
 * <p>You create chunks from text using a splitter. The splitter used for this application is the
 * {@link org.rag4j.indexing.splitters.SentenceSplitter}. In the constructor you can configure other splitters. An
 * example is the {@link org.rag4j.indexing.splitters.MaxTokenSplitter} or the {@link org.rag4j.indexing.splitters.SemanticSplitter}</p>
 * <p>The {@link org.rag4j.rag.retrieval.strategies.TopNRetrievalStrategy} is used to create the context from the
 * N most relevant chunks. Other retrieval strategies are available. You can configure the {@link org.rag4j.rag.retrieval.strategies.WindowRetrievalStrategy}
 * or the {@link org.rag4j.rag.retrieval.strategies.DocumentRetrievalStrategy} in the constructor.</p>
 * <p>You can run the class, change the questions, play around with all the options mentioned before. If you prefer to
 * use OpenAI, check the Rag4j project. It contain numerous samples to change this code into a working application
 * making use of OpenAI. Another option is to use a persistent vector store. AN implementation for Weaviate is available
 * at the project as well.</p>
 */
public class AppRAG {
    private static final Logger LOGGER = LoggerFactory.getLogger(AppRAG.class);

    private final InternalContentStore contentStore;
    private final ChatService chatService;
    private final RetrievalStrategy retrievalStrategy;

    /**
     * Main method to run the application. Read the JavaDoc for the class to learn about the different options. You can
     * change the questions to ask in this method.
     * @param args None
     */
    public static void main(String[] args) {

        AppRAG appRAG = new AppRAG();

        // Run retriever
        appRAG.askQuestion("What is RAG?", 2);
        appRAG.askQuestion("Who talked about RAG?", 2);

        // Generate an answer
        appRAG.answerQuestion("What is RAG?", 2);
        appRAG.answerQuestion("Who talked about RAG?", 2);

        // Verify quality of the answer
        appRAG.answerQuestionObserved("What is RAG?", 2);
        appRAG.answerQuestionObserved("Who talked about RAG?", 2);

    }

    /**
     * Constructor to create the application. The constructor creates the content store, chat service, and retrieval
     * strategy. The data is ingested from a file. The file is a JSONL file with the data. The splitter is used to
     * create chunks from the text. Refer to the class JavaDoc for more information about the different options.
     */
    public AppRAG() {
        OllamaAccess ollamaAccess = new OllamaAccess();
        Embedder embedder = new OllamaEmbedder(ollamaAccess);
        this.contentStore = new InternalContentStore(embedder);
        this.chatService = new OllamaChatService(ollamaAccess, "llama3.2");
        Retriever retriever = new ObservedRetriever(this.contentStore);
        this.retrievalStrategy = new TopNRetrievalStrategy(retriever);

        String fileName = "jfall/sessions-one.jsonl";
        Splitter splitter = new SentenceSplitter();
        ingestData(splitter, fileName);
    }

    /**
     * Uses the content store to ingest data from a file. The content reader reads the file. The indexing service
     * indexes the documents. The splitter is used to create chunks from the text. The embedder is included in the
     * content store.
     *
     * @param splitter Splitter to create chunks from the text
     * @param fileName Name of the file to ingest
     */
    public void ingestData(Splitter splitter, String fileName) {
        ContentReader contentReader = new JfallContentReader(fileName);
        IndexingService indexingService = new IndexingService(this.contentStore);
        indexingService.indexDocuments(contentReader, splitter);
    }

    /**
     * Asks a question to the RAG. The question is used to retrieve relevant chunks from the content store. The maximum
     * number of results is used to limit the number of results. The results are printed to the console.
     *
     * @param question Question to ask
     * @param maxResults Maximum number of results to retrieve
     */
    public void askQuestion(String question, int maxResults) {
        List<RelevantChunk> relevantChunks = contentStore.findRelevantChunks(question, maxResults);
        for (RelevantChunk relevantChunk : relevantChunks) {
            LOGGER.info("Document id: {}", relevantChunk.getDocumentId());
            LOGGER.info("Chunk id: {}", relevantChunk.getChunkId());
            LOGGER.info("Text: {}", relevantChunk.getText());
            LOGGER.info("Score: {}", relevantChunk.getScore());
            logSeparator();
        }
    }

    /**
     * Answers a question using the LLM. The question is used to retrieve relevant chunks from the content store. The
     * maximum number of results is used to limit the number of results. The answer is generated using the chat service.
     * The answer is printed to the console.
     *
     * @param question Question to ask
     * @param maxResults Maximum number of results to retrieve
     */
    public void answerQuestion(String question, int maxResults) {
        AnswerGenerator answerGenerator = new AnswerGenerator(this.chatService);
        retrieveAnswer(answerGenerator, question, maxResults);
        logSeparator();
    }

    /**
     * The same as the {@link #answerQuestion(String, int)} method. The difference is that the RAGObserver is used to
     * track the activity of the RAG system. The AnswerQualityService is used to determine the quality of the answer in
     * relation to the question and the provided context. Everything is printed to the console.
     *
     * @param question Question to ask
     * @param maxResults Maximum number of results to retrieve
     */
    public void answerQuestionObserved(String question, int maxResults) {
        ObservedAnswerGenerator answerGenerator = new ObservedAnswerGenerator(chatService);
        retrieveAnswer(answerGenerator, question, maxResults);
        RAGObserver observer = RAGTracker.getRAGObserver();
        RAGTracker.cleanup();

        RAGObserverPersistor persistor = new LoggingRAGObserverPersistor();
        persistor.persist(observer);

        AnswerQualityService answerQuality = new AnswerQualityService(chatService);
        AnswerQuality quality = answerQuality.determineQualityOfAnswer(observer);
        LOGGER.info("Quality of answer compared to the question: {}, Reason: {}}",
                quality.getAnswerToQuestionQuality().getQuality(), quality.getAnswerToQuestionQuality().getReason());
        LOGGER.info("Quality of answer coming from the context: {}, Reason {}}",
                quality.getAnswerFromContextQuality().getQuality(), quality.getAnswerFromContextQuality().getReason());
        logSeparator();
    }

    /**
     * Retrieves the answer from the chat service. The answer is generated using the answer generator. The question is
     * used to retrieve the context from the retrieval output. The context is used to generate the answer. The question
     * and answer are printed to the console.
     *
     * @param answerGenerator Answer generator to generate the answer
     * @param question Question to ask
     * @param maxResults Maximum number of results to retrieve
     */
    private void retrieveAnswer(AnswerGenerator answerGenerator, String question, int maxResults) {
        RetrievalOutput retrievalOutput = this.retrievalStrategy.retrieve(question, maxResults);
        String answer = answerGenerator.generateAnswer(question, retrievalOutput.constructContext());
        LOGGER.info("Question: {}", question);
        LOGGER.info("Answer: {}", answer);
    }

    private void logSeparator() {
        LOGGER.info("---------------------------------------");
    }
}
