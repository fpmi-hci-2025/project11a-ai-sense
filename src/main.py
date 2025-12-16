import fastapi
import pydantic
import uvicorn



class ComposeRequest(pydantic.BaseModel):

	query: str = pydantic.Field(
		...,
		description = "User's query",
		json_schema_extra = {
			"example": "Compose an article about today's news"
		}
	)

	metadata: dict | None = pydantic.Field(
		...,
		description = "Additional metadata",
		json_schema_extra = {
			"example": {
				"language": "russian",
				"tone": "friendly",
				"size": 1000
			}
		}
	)

class ComposeResponse(pydantic.BaseModel):

	article: str = pydantic.Field(
		...,
		description = "Textual article",
	)



class RecommendRequest(pydantic.BaseModel):

	id: str = pydantic.Field(
		...,
		description = "User's identifier",
		json_schema_extra = {
			"example": "user_9A9iJ22woD36h2fD"
		}
	)

class RecommendResponse(pydantic.BaseModel):

	articles: list[str] = pydantic.Field(
		...,
		description = "List of relevant articles"
	)



class Application:

	def __init__(self):

		self.application = fastapi.FastAPI(
			title = "AI API",
			version = "1.0.0",
			description = "AI Ecosystem API"
		)

	def setup(self):

		@self.application.get("/api/health")
		async def health_endpoint():

			return "OK"

		@self.application.post("/api/compose")
		async def compose_endpoint(request: ComposeRequest):

			return ComposeResponse(article = "Article example")

		@self.application.post("/api/recommend")
		async def recommend_endpoint(request: RecommendRequest):

			return RecommendResponse(articles = ["art_5DJ10jg6Ln7oN0", "art_9jHv5jNgH6UbF1"])


	def deploy(self):

		uvicorn.run(self.application, host = "0.0.0.0", port = 8000)


if __name__ == "__main__":

	application = Application()
	application.setup()
	application.deploy()
