# OpenHands Integration Report: Connecting ToM-SWE with OpenHands Agent

## Executive Summary

This report analyzes the best approaches for integrating the ToM-SWE (Theory of Mind for Software Engineering) RAG/ToM agent with the OpenHands AI agent platform. Based on research into OpenHands architecture and current integration patterns, we recommend implementing an **MCP (Model Context Protocol) server** as the primary integration approach, with a **REST API server** as a secondary option.

## OpenHands Architecture Overview

### Core Components
- **Backend**: FastAPI-based Python server handling HTTP requests and WebSocket connections
- **Frontend**: React-based UI with real-time communication via Socket.IO WebSockets
- **Agent System**: Autonomous AI agents that can modify code, run commands, browse web, and call APIs
- **Runtime Environment**: Docker-based isolated execution environments
- **Communication**: Event-driven architecture with WebSocket for real-time updates

### Current Integration Capabilities
OpenHands supports multiple integration methods:
1. **Model Context Protocol (MCP)** - Primary recommended approach
2. **REST API endpoints** - For external service integration
3. **WebSocket connections** - For real-time communication
4. **GitHub Actions** - For automated workflows
5. **CLI and headless modes** - For programmatic access

## Recommended Integration Approaches

### 1. MCP Server Integration (Primary Recommendation)

#### Why MCP is Ideal
- **Native Support**: OpenHands has built-in MCP support as documented in their official docs
- **Standardized Protocol**: Based on open standard at modelcontextprotocol.io
- **Tool Registration**: Automatic tool discovery and registration with the agent
- **Real-time Communication**: Supports both SSE (Server-Sent Events) and stdio protocols
- **Seamless Integration**: Tools appear as native capabilities to the OpenHands agent

#### Implementation Architecture
```
┌─────────────────┐    MCP Protocol    ┌─────────────────┐
│   OpenHands     │◄──────────────────►│   ToM-SWE       │
│   Agent         │                    │   MCP Server    │
│                 │                    │                 │
│ - Tool Registry │                    │ - User Analysis │
│ - Agent Logic   │                    │ - RAG Queries   │
│ - WebSocket API │                    │ - ToM Insights  │
└─────────────────┘                    └─────────────────┘
```

#### MCP Server Configuration Options

**Option A: SSE (Server-Sent Events) Server**
```toml
[mcp]
sse_servers = [
    {url="https://tom-swe-server.com/mcp", api_key="your-api-key"}
]
```

**Option B: Stdio Server (Local Process)**
```toml
[mcp]
stdio_servers = [
    {
        name="tom-swe",
        command="python",
        args=["-m", "tom_swe.mcp_server"],
        env={"LITELLM_API_KEY": "your-key"}
    }
]
```

#### Required MCP Tools to Implement
1. **`analyze_user_behavior`** - Analyze user interaction patterns
2. **`get_user_mental_state`** - Retrieve current user mental state analysis
3. **`predict_user_intent`** - Predict user's next likely actions
4. **`get_user_preferences`** - Retrieve user coding preferences and patterns
5. **`search_similar_sessions`** - RAG-based search for similar user sessions
6. **`generate_user_insights`** - Generate actionable insights about user behavior

### 2. REST API Server Integration (Secondary Option)

#### Architecture
```
┌─────────────────┐    HTTP/REST API   ┌─────────────────┐
│   OpenHands     │◄──────────────────►│   ToM-SWE       │
│   Agent         │                    │   API Server    │
│                 │                    │                 │
│ - HTTP Client   │                    │ - FastAPI       │
│ - Tool Wrapper  │                    │ - User Analysis │
│ - Agent Logic   │                    │ - RAG Endpoints │
└─────────────────┘                    └─────────────────┘
```

#### Key API Endpoints to Implement
```python
# FastAPI server endpoints
POST /api/v1/analyze-user-session
GET  /api/v1/user/{user_id}/mental-state
GET  /api/v1/user/{user_id}/preferences
POST /api/v1/search-similar-sessions
GET  /api/v1/user/{user_id}/insights
POST /api/v1/predict-intent
```

#### Integration Method
- Create custom OpenHands tools that make HTTP requests to ToM-SWE API
- Implement authentication and error handling
- Cache responses for performance optimization

## Implementation Roadmap

### Phase 1: MCP Server Development (Weeks 1-2)
1. **Set up MCP server infrastructure**
   - Choose between SSE or stdio protocol based on deployment needs
   - Implement basic MCP protocol handlers
   - Set up tool registration and discovery

2. **Implement core ToM-SWE tools**
   - `analyze_user_behavior` - Basic user pattern analysis
   - `get_user_mental_state` - Current state retrieval
   - `predict_user_intent` - Intent prediction

3. **Testing and validation**
   - Test MCP server with OpenHands locally
   - Validate tool registration and execution
   - Performance testing

### Phase 2: Advanced Features (Weeks 3-4)
1. **Enhanced RAG capabilities**
   - `search_similar_sessions` - Semantic search implementation
   - `generate_user_insights` - Advanced analytics

2. **Real-time features**
   - WebSocket support for live user analysis
   - Streaming responses for long-running analyses

3. **Production deployment**
   - Containerization with Docker
   - SSL/HTTPS configuration
   - Monitoring and logging

### Phase 3: REST API Fallback (Week 5)
1. **Implement REST API server** (if MCP has limitations)
2. **Create OpenHands tool wrappers** for API endpoints
3. **Performance optimization** and caching

## Technical Considerations

### Authentication & Security
- **API Keys**: Use environment variables for sensitive credentials
- **HTTPS**: Mandatory for production deployments
- **Rate Limiting**: Implement to prevent abuse
- **Input Validation**: Sanitize all inputs to prevent injection attacks

### Performance & Scalability
- **Caching**: Implement Redis/memory caching for frequent queries
- **Async Processing**: Use async/await for non-blocking operations
- **Database Optimization**: Optimize queries for user data retrieval
- **Load Balancing**: Consider multiple server instances for high load

### Data Privacy
- **User Consent**: Ensure user data analysis complies with privacy policies
- **Data Anonymization**: Remove PII from analysis where possible
- **Retention Policies**: Implement data cleanup procedures

## Deployment Options

### Option 1: Cloud Deployment (Recommended)
- **Platform**: AWS/GCP/Azure with container orchestration
- **Benefits**: Scalability, reliability, managed services
- **Considerations**: Cost, complexity, vendor lock-in

### Option 2: Self-Hosted
- **Platform**: Docker containers on dedicated servers
- **Benefits**: Full control, cost-effective for small scale
- **Considerations**: Maintenance overhead, security responsibility

### Option 3: Hybrid
- **Platform**: MCP server in cloud, data processing on-premises
- **Benefits**: Balance of control and scalability
- **Considerations**: Network latency, complexity

## Success Metrics

### Technical Metrics
- **Response Time**: < 500ms for basic queries, < 2s for complex analysis
- **Availability**: 99.9% uptime
- **Error Rate**: < 1% of requests
- **Tool Registration**: 100% success rate in OpenHands

### User Experience Metrics
- **Agent Enhancement**: Measurable improvement in OpenHands agent effectiveness
- **User Satisfaction**: Positive feedback on personalized assistance
- **Adoption Rate**: Usage frequency of ToM-SWE tools by OpenHands agents

## Conclusion

The **MCP server approach** is strongly recommended as the primary integration method due to:
1. Native OpenHands support and documentation
2. Standardized protocol ensuring compatibility
3. Seamless tool integration experience
4. Future-proof architecture aligned with OpenHands roadmap

The implementation should begin with a basic MCP server providing core ToM-SWE functionality, then expand to include advanced RAG capabilities and real-time features. A REST API server can serve as a backup option if specific requirements cannot be met through MCP.

This integration will enable OpenHands agents to leverage user behavior analysis and theory of mind insights, significantly enhancing their ability to provide personalized and contextually appropriate assistance to developers.