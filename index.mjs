// Import dependencies
import express from "express";
import * as dotenv from "dotenv";
dotenv.config();
import { MongoClient } from "mongodb";
import { ChatOpenAI } from "@langchain/openai";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import cors from 'cors';
import { createOpenAIFunctionsAgent, AgentExecutor } from "langchain/agents";
import { createRetrieverTool } from "langchain/tools/retriever";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { OpenAIEmbeddings } from "@langchain/openai";
import { z } from "zod";

// Express setup
const app = express();
const port = process.env.PORT || 3000;
app.use(express.json());
app.use(cors({
  origin: 'http://localhost:3001', // Replace with your frontend URL
  credentials: true // Allow sending cookies
}));

// MongoDB setup
const mongoConnectionString = process.env.MONGO_URI;
if (!mongoConnectionString) {
  throw new Error("Please add your Mongo URI to .env.local");
}
const mongoClient = new MongoClient(mongoConnectionString);
const mongoClientPromise = mongoClient.connect();

// Custom OutputParser to handle comparison and normal formats
class ManifestoOutputParser {
  parseOutput(output) {
    try {
      const parsedResponse = JSON.parse(output);

      if (parsedResponse.type === "Comparison") {
        // Ensure it fits the Comparison schema
        if (
          parsedResponse.ComparisonArray &&
          Array.isArray(parsedResponse.ComparisonArray) &&
          parsedResponse.ComparisonArray.length > 0
        ) {
          return parsedResponse;
        } else {
          return {
            type: "fallback",
            output: "The comparison did not retrieve enough details. Please try a different query."
          };
        }
      } else if (parsedResponse.type === "normal") {
        // Ensure it fits the Normal schema
        if (parsedResponse.output && typeof parsedResponse.output === "string") {
          return parsedResponse;
        }
      }
    } catch (error) {
      console.error("Failed to parse response: ", error);
      return { type: "fallback", output: "Failed to retrieve a valid response." };
    }

    // If neither condition is satisfied, return fallback
    return {
      type: "fallback",
      output: output,
    };
  }
}

// Helper function to convert the custom format history into agent-friendly message format
function formatConversationHistory(history) {
  return history.map((message) => {
    if (message.type === 'Comparison') {
      return {
        role: 'assistant',
        content: `Comparison: ${JSON.stringify(message.ComparisonArray, null, 2)}`
      };
    } else if (message.type === 'normal') {
      return {
        role: 'assistant',
        content: message.output
      };
    } else if (message.role === 'human') {
      return {
        role: 'human',
        content: message.input
      };
    }
    return null;
  }).filter(msg => msg !== null); // Filter out any null values
}

// Setup Vector Store and Agent
async function initializeAgent() {
  const mongoDbClient = await mongoClientPromise;
  const dbName = "docs";
  const nppManifestoCollectionName = "embeddingsNPP";
  const ranilManifestoCollectionName = "embeddingsRanil";
  const sajithManifestoCollectionName = "embeddingsSajith";
  const nppCollection = mongoDbClient.db(dbName).collection(nppManifestoCollectionName);
  const ranilCollection = mongoDbClient.db(dbName).collection(ranilManifestoCollectionName);
  const sajithCollection = mongoDbClient.db(dbName).collection(sajithManifestoCollectionName);

  // Vector Store Setup
  const nppVectorStore = new MongoDBAtlasVectorSearch(
    new OpenAIEmbeddings({
      modelName: "text-embedding-ada-002",
      stripNewLines: true,
    }),
    {
      collection: nppCollection,
      indexName: "default",
      textKey: "text",
      embeddingKey: "embedding",
    }
  );
  const ranilVectorStore = new MongoDBAtlasVectorSearch(
    new OpenAIEmbeddings({
      modelName: "text-embedding-ada-002",
      stripNewLines: true,
    }),
    {
      collection: ranilCollection,
      indexName: "default",
      textKey: "text",
      embeddingKey: "embedding", 
    }
  );
  const sajithVectorStore = new MongoDBAtlasVectorSearch(
    new OpenAIEmbeddings({
      modelName: "text-embedding-ada-002",
      stripNewLines: true,
    }),
    {
      collection: sajithCollection,
      indexName: "default",
      textKey: "text",
      embeddingKey: "embedding",
    }
  );

  // Increase fetchK and adjust lambda for better results
  const nppRetriever = nppVectorStore.asRetriever({
    searchType: "mmr",
    searchKwargs: { fetchK: 50, lambda: 0.2 },
    metadata: true,
  });
  const ranilRetriever = ranilVectorStore.asRetriever({
    searchType: "mmr",
    searchKwargs: { fetchK: 50, lambda: 0.2 },
    metadata: true,
  });
  const sajithRetriever = sajithVectorStore.asRetriever({
    searchType: "mmr",
    searchKwargs: { fetchK: 50, lambda: 0.2 },
    metadata: true,
  });

  // Agent Setup
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
  const nppTool = createRetrieverTool(nppRetriever, {
    name: "NPP_Manifesto_Search",
    description: "Search for information from the National People's Power (NPP) manifesto.",
  });
  const ranilTool = createRetrieverTool(ranilRetriever, {
    name: "Ranil_Wickremesinghe_Manifesto_Search",
    description: "Search for information from Ranil Wickremesinghe's manifesto.",
  });
  const sajithTool = createRetrieverTool(sajithRetriever, {
    name: "Sajith_Premadasa_Manifesto_Search",
    description: "Search for information from Sajith Premadasa's manifesto.",
  });

  // Create agent instance
  const agentInstance = await createOpenAIFunctionsAgent({
    llm: chatModel,
    prompt: agentPromptTemplate,
    tools: [nppTool, ranilTool, sajithTool],
  });

  // Create agent executor
  const agentExecutor = new AgentExecutor({
    agent: agentInstance,
    tools: [nppTool, ranilTool, sajithTool],
  });

  return agentExecutor;
}

// API endpoint for handling user queries
app.post("/query", async (req, res) => {
  const { input, chat_history } = req.body;
  if (!input || !Array.isArray(chat_history)) {
    return res
      .status(400)
      .json({ error: "Input and chat history are required" });
  }

  const agentExecutor = await initializeAgent();
  const outputParser = new ManifestoOutputParser();
  
  try {
    // Format chat history before sending it to the agent
    const formattedHistory = formatConversationHistory(chat_history);

    // Log agent's raw response for debugging
    const agentResponse = await agentExecutor.invoke({
      input: input,
      chat_history: formattedHistory,  // Using formatted chat history here
    });
    
    // console.log("Raw agent response:", agentResponse); // Log the raw response

    // Parse the output using the custom parser
    const parsedAgentResponse = outputParser.parseOutput(agentResponse.output);

    // Add new parsed response to chat history for further queries
    const updatedChatHistory = [
      ...chat_history,
      { role: "human", input },
      parsedAgentResponse,
    ];

    res.json({ response: parsedAgentResponse, chat_history: updatedChatHistory });
  } catch (error) {
    console.error("Error processing query:", error);
    res.status(500).json({ error: "An error occurred while processing the query" });
  }
});

// Start the server
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
