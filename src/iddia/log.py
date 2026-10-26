"""Çalıştırma uyarıları: API kesintileri, atlanan adımlar.

Her uyarı konsola yazılır ve sitede "Son çalıştırma" altında gösterilir; böylece
bir tahminin hangi veriler eksikken üretildiği sonradan görülebilir.
"""
WARNINGS: list[dict] = []


def warn(source: str, message: str) -> None:
    print(f"{source}: {message}")
    WARNINGS.append({"kaynak": source, "mesaj": message})
