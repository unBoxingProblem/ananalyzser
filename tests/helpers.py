from io import BytesIO


class Upload(BytesIO):
    def __init__(self, data: bytes, name: str, mime: str = "application/octet-stream"):
        super().__init__(data)
        self.name = name
        self.type = mime

    @property
    def size(self) -> int:
        return len(self.getbuffer())
