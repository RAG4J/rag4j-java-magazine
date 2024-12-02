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
import org.rag4j.rag.generation.QuestionGenerator;
import org.rag4j.rag.generation.QuestionGeneratorService;
import org.rag4j.rag.generation.chat.ChatService;
import org.rag4j.rag.generation.quality.AnswerQuality;
import org.rag4j.rag.generation.quality.AnswerQualityService;
import org.rag4j.rag.model.RelevantChunk;
import org.rag4j.rag.retrieval.ObservedRetriever;
import org.rag4j.rag.retrieval.RetrievalOutput;
import org.rag4j.rag.retrieval.RetrievalStrategy;
import org.rag4j.rag.retrieval.Retriever;
import org.rag4j.rag.retrieval.quality.QuestionAnswerRecord;
import org.rag4j.rag.retrieval.quality.RetrievalQuality;
import org.rag4j.rag.retrieval.quality.RetrievalQualityService;
import org.rag4j.rag.retrieval.strategies.DocumentRetrievalStrategy;
import org.rag4j.rag.retrieval.strategies.TopNRetrievalStrategy;
import org.rag4j.rag.store.local.InternalContentStore;
import org.rag4j.rag.tracker.LoggingRAGObserverPersistor;
import org.rag4j.rag.tracker.RAGObserver;
import org.rag4j.rag.tracker.RAGObserverPersistor;
import org.rag4j.rag.tracker.RAGTracker;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.List;

import static java.lang.System.getProperty;

/**
 * <p>This is an application that demonstrates the use of the RAG library RAG4J. More information about the framework is
 * available at the <a href="https://rag4j.github.io/rag4j/">project's GitHub page</a>. <a href="TODO">A blog post</a> is
 * explaining the steps to create this application. If you just want to run it, you came to the right place. You can
 * make some changes if you want to use other Splitters, strategies or Ollama models.</p>
 * <p>There are two data files. You start with sessions-one.jsonl. This is faster to try different splitters. The
 * second dataset contains all the talks from the jFall conference 2024. Some backups are available for the bigger
 * dataset.</p>
 * <p>In this example, you use Ollama. Ollama runs different LLMs on you machine. You can use multiple models. Beware
 * that some models work better than others. This project is tested with the <strong>llama3.2</strong> model. You can
 * configure other models. Check the <a href="https://ollama.com/search">Ollama website</a> for more information. </p>
 * <p>For this application you use the {@link org.rag4j.rag.store.local.InternalContentStore}. This class uses a Map
 * to store the vectors and chunks. The store requires an embedder to create embeddings from the chunks you want to
 * store. You use the default Ollama model for the rag4j framework: <strong>nomic-embed-text</strong>.</p>
 * <p>You create chunks from text using a splitter. The splitter used for this application is the
 * {@link org.rag4j.indexing.splitters.SentenceSplitter}. In step 1 you can configure other splitters. An
 * example is the {@link org.rag4j.indexing.splitters.MaxTokenSplitter} or the {@link org.rag4j.indexing.splitters.SemanticSplitter}</p>
 * <p>The {@link org.rag4j.rag.retrieval.strategies.TopNRetrievalStrategy} is used to create the context from the
 * N most relevant chunks. Other retrieval strategies are available. You can configure the {@link org.rag4j.rag.retrieval.strategies.WindowRetrievalStrategy}
 * or the {@link org.rag4j.rag.retrieval.strategies.DocumentRetrievalStrategy} in step 2.</p>
 * <p>You can run the class, change the questions, play around with all the options mentioned before. If you prefer to
 * use OpenAI, check the Rag4j project. It contains numerous samples to change this code into a working application
 * making use of OpenAI. Another option is to use a persistent vector store. An implementation for Weaviate is available
 * at the project as well.</p>
 */
public class AppRAG {
    private static final Logger LOGGER = LoggerFactory.getLogger(AppRAG.class);
    private static final String DEFAULT_OLLAMA_MODEL = "llama3.2";

    private final OllamaAccess ollamaAccess;
    private final InternalContentStore contentStore;
    private final Retriever retriever;
    private final Embedder embedder;

    private ChatService chatService;
    private RetrievalStrategy retrievalStrategy;


    /**
     * Main method to run the application. Read the JavaDoc for the class to learn about the different options. You can
     * change the questions to ask in this method.
     *
     * @param args None
     */
    public static void main(String[] args) {

        AppRAG appRAG = new AppRAG();

        // Step 1: Ingest data or load from backup
        appRAG.ingestData(new SentenceSplitter(), "jfall/sessions-one.jsonl");
        // OR load backup from disk
//         appRAG.loadBackup("ollama-maxtoken-100-all");

        // Step 1a (Optional): Store backup, check the log for the path to the backup file.
//        appRAG.storeBackup("ollama-maxtoken-100-all");

        // Step 2: Retrieve related chunks
        appRAG.retrieveRelatedChunks("What is RAG?", 2);
        appRAG.retrieveRelatedChunks("Who talked about RAG?", 2);

        // Step 2a: Generate a judgement list
        Path judgementListPath = appRAG.generateJudgementList("jfall_questions_answers_sample.csv");

        // Step 2b: Determine the quality of the retriever using the judgement list, use the available judgement list
        //  when using a backup for the data
//        Path judgementListPath = Path.of(getProperty("user.dir"), "backups","jfall_judgment-ollama-maxtoken-100-all.csv");
        appRAG.runJudgementList(judgementListPath);

        // Step 3: Generate an answer. Optionally, modify the chat service or retrieval strategy
//        appRAG.modifyChatService("llama3.2");
//        appRAG.modifyRetrievalStrategy(new DocumentRetrievalStrategy(appRAG.getRetriever()));
//        appRAG.answerQuestion("What is RAG?", 2);
//        appRAG.answerQuestion("Who talked about RAG?", 2);

        // Step 4: Generate an answer and track the quality of the RAG system
//        appRAG.answerQuestionObserved("What is RAG?", 2);
//        appRAG.answerQuestionObserved("Who talked about RAG?", 2);
    }

    /**
     * Constructor to create the application. The constructor creates the content store, chat service, and retrieval
     * strategy. The data is ingested from a file. The file is a JSONL file with the data. The splitter is used to
     * create chunks from the text. Refer to the class JavaDoc for more information about the different options.
     */
    public AppRAG() {
        this.ollamaAccess = new OllamaAccess();
        this.embedder = new OllamaEmbedder(ollamaAccess);
        this.contentStore = new InternalContentStore(embedder);
        this.retriever = new ObservedRetriever(this.contentStore);

        // Setting some defaults
        this.chatService = new OllamaChatService(ollamaAccess, DEFAULT_OLLAMA_MODEL);
        this.retrievalStrategy = new TopNRetrievalStrategy(retriever);
    }

    /**
     * Loads the backup from disk. The backup is stored in the backups folder. The backup name is used to load the
     * backup. The backup consists of two parts, the data in the metadata in separate files.
     * @param backupName Name of the backup to load
     */
    public void loadBackup(String backupName) {
        Path backUpPath = Path.of(System.getProperty("user.dir"), "/backups");
        this.contentStore.loadFromDisk(backUpPath, backupName);
    }

    /**
     * Creates a backup from the content store. The backup is stored in the backups folder. The backup contains two
     * parts, the data and the metadata.
     * @param backupName String to use as the name of the backup
     */
    public void storeBackup(String backupName) {
        Path backUpPath = Path.of(System.getProperty("user.dir"), "/backups");
        this.contentStore.backupToDisk(backUpPath, backupName);
    }

    /**
     * Modifies the chat service to use a different model. The model is used to generate the answer. Make sure the
     * model is available in the Ollama system.
     * @param model String representing the model to use
     */
    public void modifyChatService(String model) {
        this.chatService = new OllamaChatService(this.ollamaAccess, model);
    }

    /**
     * Modifies the retrieval strategy to use a different strategy. The strategy is used to retrieve relevant chunks
     * from the content store. You can use the {@link AppRAG#getRetriever()} method to get the retriever to use in the
     * strategy.
     * @param strategy Retrieval strategy to use
     */
    public void modifyRetrievalStrategy(RetrievalStrategy strategy) {
        this.retrievalStrategy = strategy;
    }

    /**
     * Returns the observed retriever. The retriever is used to retrieve relevant chunks from the content store.
     * @return Retriever to retrieve relevant chunks
     */
    public Retriever getRetriever() {
        return this.retriever;
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
     * Asks a question to the LLM. The question is used to retrieve relevant chunks from the content store. The maximum
     * number of results is used to limit the number of results. The results are printed to the console.
     *
     * @param question   Question to ask
     * @param maxResults Maximum number of results to retrieve
     */
    public void retrieveRelatedChunks(String question, int maxResults) {
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
     * @param question   Question to ask
     * @param maxResults Maximum number of results to retrieve
     */
    public void answerQuestion(String question, int maxResults) {
        AnswerGenerator answerGenerator = new AnswerGenerator(this.chatService);
        retrieveAnswer(answerGenerator, question, maxResults);
        logSeparator();
    }

    /**
     * Generates a Judgment List. The Judgment List is used to determine the quality of the retrieval system. The
     * generator is used to generate a question for each chunk in the content store. The question and answer pairs are
     * saved to a file.
     *
     * @param fileName Name of the file to save the question and answer pairs
     */
    public Path generateJudgementList(String fileName) {
        QuestionGenerator questionGenerator = new QuestionGenerator(this.chatService);
        QuestionGeneratorService questionGeneratorService =
                new QuestionGeneratorService(this.contentStore, questionGenerator);
        Path savedFilePath = questionGeneratorService.generateQuestionAnswerPairsAndSaveToTempFile(fileName);
        LOGGER.info("Saved file: {}", savedFilePath);
        return savedFilePath;
    }

    public void runJudgementList(Path fileName) {
        ObservedRetriever observedRetriever = new ObservedRetriever(this.contentStore);
        RetrievalQualityService retrievalQualityService = new RetrievalQualityService(observedRetriever);
        List<QuestionAnswerRecord> questionAnswerRecords =
                retrievalQualityService.readQuestionAnswersFromFilePath(fileName, false);
        RetrievalQuality retrievalQuality =
                retrievalQualityService.obtainRetrievalQuality(questionAnswerRecords, this.embedder);

        LOGGER.info("Correct: {}", retrievalQuality.getCorrect());
        LOGGER.info("Incorrect: {}", retrievalQuality.getIncorrect());
        LOGGER.info("Quality using precision: {}", retrievalQuality.getPrecision());
        LOGGER.info("Total questions: {}", retrievalQuality.totalItems());
        this.logSeparator();
    }

    /**
     * The same as the {@link #answerQuestion(String, int)} method. The difference is that the RAGObserver is used to
     * track the activity of the RAG system. The AnswerQualityService is used to determine the quality of the answer in
     * relation to the question and the provided context. Everything is printed to the console.
     *
     * @param question   Question to ask
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
     * @param question        Question to ask
     * @param maxResults      Maximum number of results to retrieve
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
