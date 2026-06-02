from config import settings

class SmartRouter:
    @staticmethod
    def route(metrics: dict) -> str:
        skin_ratio = metrics.get("skin_ratio", 0.0)
        graphic_score = metrics.get("graphic_score", 0.0)
        coverage = metrics.get("coverage", 0.0)
        edge_density = metrics.get("edge_density", 0.0)
        complexity = metrics.get("complexity", 0.0)

        # 1. Definitive Portrait Override
        # Ensure images with clear human subjects (skin_ratio > 0.35) 
        # go to portrait route, ignoring moderate graphic_scores.
        if skin_ratio > 0.35:
            print(f"[Router] Override: High skin_ratio ({skin_ratio:.4f}) -> ROUTE_PORTRAIT")
            return "ROUTE_PORTRAIT"

        # 2. Weighted Graphic Scoring
        # Penalize graphic confidence if there are natural human features (skin) or high edge density.
        graphic_confidence = graphic_score - (skin_ratio * 100.0)
        
        # 3. Decision Tree
        if graphic_confidence > settings.graphic_asset_threshold:
            print(f"[Router] Graphic confidence ({graphic_confidence:.1f}) -> ROUTE_GRAPHIC")
            return "ROUTE_GRAPHIC"

        if skin_ratio > settings.portrait_skin_threshold:
            print(f"[Router] Moderate skin_ratio ({skin_ratio:.4f}) -> ROUTE_PORTRAIT")
            return "ROUTE_PORTRAIT"
            
        if complexity > settings.sam2_complexity_threshold or coverage > settings.sam2_coverage_threshold:
            print(f"[Router] High complexity/coverage -> ROUTE_COMPLEX")
            return "ROUTE_COMPLEX"
            
        return "ROUTE_SIMPLE"
