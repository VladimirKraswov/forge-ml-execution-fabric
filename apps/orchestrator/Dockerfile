# apps/orchestrator/Dockerfile
FROM node:20-slim
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 18787
CMD ["node", "src/server.js"]