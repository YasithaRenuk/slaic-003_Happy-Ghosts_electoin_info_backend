import express from "express";
import * as dotenv from "dotenv";
dotenv.config();
import { MongoClient } from "mongodb";
import { ChatOpenAI } from "@langchain/openai";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import cors from "cors";
import { createOpenAIFunctionsAgent, AgentExecutor } from "langchain/agents";
import { createRetrieverTool } from "langchain/tools/retriever";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { OpenAIEmbeddings } from "@langchain/openai";

// Express setup
const app = express();
const port = process.env.PORT || 3000;
app.use(express.json());
app.use(
  cors({
    origin: "http://localhost:3001", // Replace with your frontend URL
    credentials: true, // Allow sending cookies
  })
);

// MongoDB setup
const mongoConnectionString = process.env.MONGO_URI;
if (!mongoConnectionString) {
  throw new Error("Please add your Mongo URI to .env.local");
}
const mongoClient = new MongoClient(mongoConnectionString);
const mongoClientPromise = mongoClient.connect().catch((error) => {
  console.error("Error connecting to MongoDB:", error);
  process.exit(1);
});

// Custom OutputParser to handle comparison and normal formats
class ManifestoOutputParser {
  parseOutput(output) {
    try {
      const parsedResponse = JSON.parse(output);
      const { type, output: normalOutput, ComparisonArray } = parsedResponse;

      if (
        type === "Comparison" &&
        Array.isArray(ComparisonArray) &&
        ComparisonArray.length > 0
      ) {
        return parsedResponse;
      } else if (type === "normal" && typeof normalOutput === "string") {
        return parsedResponse;
      }
    } catch (error) {
      console.error("Failed to parse response:", error);
    }

    return { type: "fallback", output: "Failed to retrieve a valid response." };
  }
}

// Helper function to format conversation history
function formatConversationHistory(history) {
  return history
    .map((message) => {
      if (message.type === "Comparison") {
        return {
          role: "assistant",
          content: `Comparison: ${JSON.stringify(
            message.ComparisonArray,
            null,
            2
          )}`,
        };
      } else if (message.type === "normal") {
        return { role: "assistant", content: message.output };
      } else if (message.role === "human") {
        return { role: "human", content: message.input };
      }
      return null;
    })
    .filter(Boolean);
}

// Helper function to create vector store and retriever
async function createVectorStoreAndRetriever(client, collectionName) {
  const collection = client.db("docs").collection(collectionName);
  const vectorStore = new MongoDBAtlasVectorSearch(
    new OpenAIEmbeddings({
      modelName: "text-embedding-ada-002",
      stripNewLines: true,
    }),
    {
      collection,
      indexName: "default",
      textKey: "text",
      embeddingKey: "embedding",
    }
  );
  return vectorStore.asRetriever({
    searchType: "mmr",
    searchKwargs: { fetchK: 50, lambda: 0.2 },
    metadata: true,
  });
}

// Setup Agent
async function initializeAgent() {
  try {
    const client = await mongoClientPromise;

    // Load vector stores concurrently
    const [nppRetriever, ranilRetriever, sajithRetriever] = await Promise.all([
      createVectorStoreAndRetriever(client, "embeddingsNPP"),
      createVectorStoreAndRetriever(client, "embeddingsRanil"),
      createVectorStoreAndRetriever(client, "embeddingsSajith"),
    ]);

    // Agent prompt template
    const agentPromptTemplate = ChatPromptTemplate.fromMessages([
      {
        role: "system",
        content: `"You are a well-informed assistant with access to political manifestos. Answer questions based on specific manifesto content related to the National People's Power (NPP), Ranil Wickremesinghe, or Sajith Premadasa. Structure your answers in a well-formatted JSON object. If you are comparing manifestos, use the following format:
  
  {{
    type: 'Comparison',
    title: 'Comparison between [Manifesto A] and [Manifesto B]',
    ComparisonArray: [
      {{
        name: '[Manifesto A]',
        pointArray: [
          {{ pointTitle: '[Topic 1]', point: '[Details about Manifesto A]' }},
          {{ pointTitle: '[Topic 2]', point: '[Details about Manifesto A]' }}
        ]
      }},
      {{
        name: '[Manifesto B]',
        pointArray: [
          {{ pointTitle: '[Topic 1]', point: '[Details about Manifesto B]' }},
          {{ pointTitle: '[Topic 2]', point: '[Details about Manifesto B]' }}
        ]
      }}
    ],
    keyPoints: '[Summary of the comparison]'
  }}
  
  If it's a regular answer or just a reply, structure the output as:
  {{
    type: 'normal',
    output: '[The answer]'
  }}"`,
      },
      new MessagesPlaceholder("chat_history"),
      { role: "human", content: "{input}" },
      new MessagesPlaceholder("agent_scratchpad"),
    ]);

    const chatModel = new ChatOpenAI({ modelName: "gpt-4o", temperature: 0.2 });

    // Create tools for each manifesto retriever
    const tools = [
      createRetrieverTool(nppRetriever, {
        name: "NPP_Manifesto_Search",
        description:
          "Search for information from the National People's Power (NPP) manifesto.",
      }),
      createRetrieverTool(ranilRetriever, {
        name: "Ranil_Wickremesinghe_Manifesto_Search",
        description:
          "Search for information from Ranil Wickremesinghe's manifesto.",
      }),
      createRetrieverTool(sajithRetriever, {
        name: "Sajith_Premadasa_Manifesto_Search",
        description:
          "Search for information from Sajith Premadasa's manifesto.",
      }),
    ];

    // Create agent instance
    const agentInstance = await createOpenAIFunctionsAgent({
      llm: chatModel,
      prompt: agentPromptTemplate,
      tools,
    });

    // Create agent executor
    return new AgentExecutor({ agent: agentInstance, tools });
  } catch (error) {
    console.error("Error initializing agent:", error);
    throw error;
  }
}

// API endpoint for handling user queries
app.post("/query", async (req, res) => {
  const { input, chat_history } = req.body;
  if (!input || !Array.isArray(chat_history)) {
    return res
      .status(400)
      .json({ error: "Input and chat history are required" });
  }

  try {
    const agentExecutor = await initializeAgent();
    const outputParser = new ManifestoOutputParser();

    // Format chat history
    const formattedHistory = formatConversationHistory(chat_history);

    // Execute agent
    const agentResponse = await agentExecutor.invoke({
      input,
      chat_history: formattedHistory,
    });

    // Parse agent response
    const parsedAgentResponse = outputParser.parseOutput(agentResponse.output);

    // Update chat history and return response
    const updatedChatHistory = [
      ...chat_history,
      { role: "human", input },
      parsedAgentResponse,
    ];

    res.json({
      response: parsedAgentResponse,
      chat_history: updatedChatHistory,
    });
  } catch (error) {
    console.error("Error processing query:", error);
    res
      .status(500)
      .json({ error: "An error occurred while processing the query" });
  }
});

// Start the server
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
