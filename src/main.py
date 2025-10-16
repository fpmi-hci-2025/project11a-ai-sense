import fastapi
import pydantic
import uvicorn



class HealthResponse(pydantic.BaseModel):

	status: str = "OK"


class Application:

	def __init__(self):

		self.application = fastapi.FastAPI()

	def setup(self):

		@self.application.get("/api/health")
		async def health_endpoint():

			return HealthResponse()

	def deploy(self):

		uvicorn.run(self.application, host = "0.0.0.0", port = 8000)


if __name__ == "__main__":

	application = Application()
	application.setup()
	application.deploy()
