import os

# Windows : mise à l'échelle d'affichage (125 %, 150 %…) -> sans ça, plein écran flou/mal dimensionné.
# Doit être posé avant pygame.init() ; sans effet hors Windows.
os.environ.setdefault("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2")
