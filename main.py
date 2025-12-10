# =================================  TESTY  ===================================
# Testy do tego pliku obejmują jedynie weryfikację poprawności wyników dla
# prawidłowych danych wejściowych - obsługa niepoprawych danych wejściowych
# nie jest ani wymagana ani sprawdzana. W razie potrzeby lub chęci można ją 
# wykonać w dowolny sposób we własnym zakresie.
# =============================================================================
import numpy as np


def chebyshev_nodes(n: int = 10) -> np.ndarray | None:
    """Funkcja generująca wektor węzłów Czebyszewa drugiego rodzaju (n,) 
    i sortująca wynik od najmniejszego do największego węzła.

    Args:
        n (int): Liczba węzłów Czebyszewa.
    
    Returns:
        (np.ndarray): Wektor węzłów Czebyszewa (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(n, int) or n < 1:
        return None
    k = np.arange(n)
    if n == 1:
        nodes = np.array([0.0])
    else:
        nodes = np.cos(k * np.pi / (n - 1))
    
    return nodes



def bar_cheb_weights(n: int = 10) -> np.ndarray | None:
    """Funkcja tworząca wektor wag dla węzłów Czebyszewa wymiaru (n,).

    Args:
        n (int): Liczba wag węzłów Czebyszewa.
    
    Returns:
        (np.ndarray): Wektor wag dla węzłów Czebyszewa (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(n, int) or n < 1:
        return None
    w = np.zeros(n)
    
    for j in range(n):

        if j == 0 or j == n-1:  
            delta_j = 0.5
        else:
            delta_j = 1.0

        w[j] = ((-1) ** j) * delta_j
    return w


import numpy as np

def barycentric_inte(
    xi: np.ndarray, yi: np.ndarray, wi: np.ndarray, x: np.ndarray
) -> np.ndarray | None:
    """Funkcja przeprowadza interpolację metodą barycentryczną dla zadanych 
    węzłów xi i wartości funkcji interpolowanej yi używając wag wi. Zwraca 
    wyliczone wartości funkcji interpolującej dla argumentów x w postaci 
    wektora (n,).

    Args:
        xi (np.ndarray): Wektor węzłów interpolacji (m,).
        yi (np.ndarray): Wektor wartości funkcji interpolowanej w węzłach (m,).
        wi (np.ndarray): Wektor wag interpolacji (m,).
        x (np.ndarray): Wektor argumentów dla funkcji interpolującej (n,).
    
    Returns:
        (np.ndarray): Wektor wartości funkcji interpolującej (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not (isinstance(xi, np.ndarray) and isinstance(yi, np.ndarray) and 
            isinstance(wi, np.ndarray) and isinstance(x, np.ndarray)):
        return None
    

    if xi.ndim != 1 or yi.ndim != 1 or wi.ndim != 1 or x.ndim != 1:
        return None
    
    if not (len(xi) == len(yi) == len(wi)):
        return None
    
    m = len(xi)
    n = len(x)
    
    y_interp = np.zeros(n)

    for i, x_val in enumerate(x):

        mask = np.abs(x_val - xi) < 1e-12
        if np.any(mask):

            y_interp[i] = yi[mask][0]
        else:
            diff = x_val - xi
            L = wi / diff
            y_interp[i] = np.sum(yi * L) / np.sum(L)
    
    return y_interp


def L_inf(
    xr: int | float | list | np.ndarray, x: int | float | list | np.ndarray
) -> float | None:
    """Funkcja obliczająca normę L-nieskończoność. Powinna działać zarówno na 
    wartościach skalarnych, listach, jak i wektorach biblioteki numpy.

    Args:
        xr (int | float | list | np.ndarray): Wartość dokładna w postaci 
            skalara, listy lub wektora (n,).
        x (int | float | list | np.ndarray): Wartość przybliżona w postaci 
            skalara, listy lub wektora (n,).

    Returns:
        (float): Wartość normy L-nieskończoność.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    try:

        xr_array = np.asarray(xr, dtype=float)
        x_array = np.asarray(x, dtype=float)
        

        if xr_array.shape != x_array.shape:
            return None
        

        norm = np.max(np.abs(xr_array - x_array))
        
        return float(norm)
        
    except (ValueError, TypeError):
        return None

