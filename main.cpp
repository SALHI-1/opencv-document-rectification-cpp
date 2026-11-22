/**
 * ============================================================================
 *  PROJET : Redressement et Orthorectification de Documents par Homographie 2D
 *  AUTEUR : Mohammed SALHI
 *  LANGAGE: C++17 / OpenCV 4.x
 * ============================================================================
 *  DESCRIPTION :
 *  Programme permettant de redresser la photographie d'une page posée sur un
 *  support afin d'obtenir une vue de dessus (orthorectifiée).
 *  Intègre les 4 étapes demandées par l'énoncé :
 *    1. Détection automatique personnalisée des 4 coins du document :
 *       - Fond simple (uniforme et contrasté) : Binarisation globale d'Otsu
 *         couplée à une fermeture morphologique et approximation polygonale.
 *       - Fond texturé (complexe ou bruité) : Filtrage bilatéral préservant
 *         les arêtes, gradient morphologique / Canny multi-échelle et
 *         approximation de Ramer-Douglas-Peucker sur enveloppe convexe.
 *    2. Affichage graphique du résultat de la détection (quadrilatère + sommets).
 *    3. Phase d'ajustement interactif des points détectés à la souris (Drag & Drop).
 *    4. Application en temps réel de la transformation de perspective.
 * ============================================================================
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <filesystem>
#include <set>
#include <iomanip>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// Modes de traitement selon la nature de l'arrière-plan
enum class BackgroundType {
    AUTO_DETECT,      // Classification automatique via la variance du Laplacien en bordure
    SIMPLE_UNIFORM,   // Fond simple (uniforme et contrasté par rapport à la page)
    TEXTURED_NOISY    // Fond texturé (complexe, boisé, motifs ou bruité)
};

// Contexte global de la session d'orthorectification
struct DocumentScannerContext {
    cv::Mat sourceImg;                 // Image originale non modifiée
    cv::Mat canvasImg;                 // Canevas d'affichage avec surcouche graphique
    cv::Mat rectifiedImg;              // Image redressée (vue de dessus)
    std::vector<cv::Point2f> corners;  // Coordonnées des 4 sommets du quadrilatère
    
    BackgroundType selectedMode = BackgroundType::AUTO_DETECT;
    std::string activeMethodLabel = "Initialisation";
    double borderComplexityMetric = 0.0;
    
    fs::path sourceFilePath;
    std::string fileExt = ".jpg";
    std::string fileStem = "document";
    
    int activeVertexIdx = -1;          // Sommet actuellement déplacé à la souris
    int hoveredVertexIdx = -1;         // Sommet survolé par le curseur
    
    std::string mainWinTitle = "Mohammed SALHI - Detection & Ajustement des 4 Coins";
    std::string previewWinTitle = "Mohammed SALHI - Vue de Dessus (Orthorectifiee)";
};

// ============================================================================
// 1. UTILITAIRES DE FICHIERS ET CHARGEMENT D'IMAGES
// ============================================================================

const std::set<std::string> VALID_IMAGE_FORMATS = {
    ".jpg", ".jpeg", ".jpe", ".png", ".bmp", ".dib",
    ".webp", ".tiff", ".tif", ".pbm", ".pgm", ".ppm", ".pnm"
};

std::string normalizeExt(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return text;
}

bool hasValidImageExtension(const fs::path& filePath) {
    std::string ext = normalizeExt(filePath.extension().string());
    return VALID_IMAGE_FORMATS.find(ext) != VALID_IMAGE_FORMATS.end();
}

std::string locateFirstImage(const fs::path& folder) {
    if (!fs::exists(folder) || !fs::is_directory(folder)) return "";
    
    std::vector<fs::path> candidates;
    for (const auto& item : fs::directory_iterator(folder)) {
        if (item.is_regular_file() && hasValidImageExtension(item.path())) {
            candidates.push_back(item.path());
        }
    }
    std::sort(candidates.begin(), candidates.end());
    return candidates.empty() ? "" : candidates.front().string();
}

std::string resolveTargetImage(const std::string& inputArg) {
    if (!inputArg.empty()) {
        fs::path candidate(inputArg);
        if (fs::is_regular_file(candidate)) return candidate.string();
        if (fs::is_directory(candidate)) {
            std::string img = locateFirstImage(candidate);
            if (!img.empty()) return img;
        }
        return inputArg;
    }
    
    std::string fromData = locateFirstImage("data");
    if (!fromData.empty()) return fromData;
    
    std::string fromRoot = locateFirstImage(".");
    if (!fromRoot.empty()) return fromRoot;
    
    return "data/pa.JPG";
}

// ============================================================================
// 2. ORDONNANCEMENT GÉOMÉTRIQUE ET TRANSFORMATION DE PERSPECTIVE
// ============================================================================

/**
 * @brief Ordonne canoniquement les 4 sommets du quadrilatère :
 *        [0] Haut-Gauche (TL), [1] Haut-Droit (TR),
 *        [2] Bas-Droit (BR),   [3] Bas-Gauche (BL).
 */
std::vector<cv::Point2f> sortQuadrilateralVertices(const std::vector<cv::Point2f>& rawPts) {
    if (rawPts.size() != 4) return rawPts;
    
    std::vector<cv::Point2f> sorted(4);
    
    // Tri par rapport au barycentre pour éviter toute ambiguïté lors de fortes rotations
    cv::Point2f center(0.0f, 0.0f);
    for (const auto& pt : rawPts) {
        center += pt;
    }
    center *= 0.25f;
    
    std::vector<float> sumCoords(4);
    std::vector<float> diffCoords(4);
    for (size_t i = 0; i < 4; ++i) {
        sumCoords[i]  = rawPts[i].x + rawPts[i].y;
        diffCoords[i] = rawPts[i].y - rawPts[i].x;
    }
    
    int tl = static_cast<int>(std::min_element(sumCoords.begin(), sumCoords.end()) - sumCoords.begin());
    int br = static_cast<int>(std::max_element(sumCoords.begin(), sumCoords.end()) - sumCoords.begin());
    int tr = static_cast<int>(std::min_element(diffCoords.begin(), diffCoords.end()) - diffCoords.begin());
    int bl = static_cast<int>(std::max_element(diffCoords.begin(), diffCoords.end()) - diffCoords.begin());
    
    sorted[0] = rawPts[tl];
    sorted[1] = rawPts[tr];
    sorted[2] = rawPts[br];
    sorted[3] = rawPts[bl];
    
    return sorted;
}

/**
 * @brief Applique l'homographie planaire (transformation de perspective) pour
 *        projeter le quadrilatère détecté vers un rectangle orthogonal (vue de dessus).
 */
cv::Mat computeTopDownPerspective(const cv::Mat& input, const std::vector<cv::Point2f>& quad) {
    if (quad.size() != 4) return cv::Mat();
    
    std::vector<cv::Point2f> corners = sortQuadrilateralVertices(quad);
    const cv::Point2f& tl = corners[0];
    const cv::Point2f& tr = corners[1];
    const cv::Point2f& br = corners[2];
    const cv::Point2f& bl = corners[3];
    
    // Calcul des dimensions euclidiennes du document redressé
    float topEdge    = std::hypot(tr.x - tl.x, tr.y - tl.y);
    float bottomEdge = std::hypot(br.x - bl.x, br.y - bl.y);
    int targetWidth  = static_cast<int>(std::round(std::max(topEdge, bottomEdge)));
    
    float leftEdge   = std::hypot(bl.x - tl.x, bl.y - tl.y);
    float rightEdge  = std::hypot(br.x - tr.x, br.y - tr.y);
    int targetHeight = static_cast<int>(std::round(std::max(leftEdge, rightEdge)));
    
    if (targetWidth <= 15 || targetHeight <= 15) return cv::Mat();
    
    std::vector<cv::Point2f> destinationRect = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(static_cast<float>(targetWidth - 1), 0.0f),
        cv::Point2f(static_cast<float>(targetWidth - 1), static_cast<float>(targetHeight - 1)),
        cv::Point2f(0.0f, static_cast<float>(targetHeight - 1))
    };
    
    cv::Mat homography = cv::getPerspectiveTransform(corners, destinationRect);
    cv::Mat topDownView;
    cv::warpPerspective(input, topDownView, homography, cv::Size(targetWidth, targetHeight), cv::INTER_LINEAR);
    
    return topDownView;
}

// ============================================================================
// 3. ALGORITHMES DE DÉTECTION DES 4 COINS (FOND SIMPLE & FOND TEXTURÉ)
// ============================================================================

/**
 * @brief Évalue la complexité texturale de l'arrière-plan en mesurant la
 *        variance de l'opérateur Laplacien sur les marges périphériques (10%).
 */
double evaluateBackgroundComplexity(const cv::Mat& image) {
    cv::Mat gray, lap;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Laplacian(gray, lap, CV_64F);
    
    cv::Mat marginMask = cv::Mat::zeros(gray.size(), CV_8U);
    int marginW = std::max(1, image.cols / 10);
    int marginH = std::max(1, image.rows / 10);
    
    marginMask(cv::Rect(0, 0, image.cols, marginH)).setTo(255);
    marginMask(cv::Rect(0, image.rows - marginH, image.cols, marginH)).setTo(255);
    marginMask(cv::Rect(0, 0, marginW, image.rows)).setTo(255);
    marginMask(cv::Rect(image.cols - marginW, 0, marginW, image.rows)).setTo(255);
    
    cv::Scalar mu, sigma;
    cv::meanStdDev(lap, mu, sigma, marginMask);
    return sigma[0] * sigma[0];
}

/**
 * @brief Vérifie qu'un quadrilatère candidat ne fusionne pas la feuille avec
 *        une bordure entière de l'image (cas où la table claire est seuillée avec la page).
 */
bool isInteriorDocumentQuad(const std::vector<cv::Point2f>& quad, const cv::Size& imgSize) {
    if (quad.size() != 4) return false;
    int borderTouches = 0;
    const float tol = 4.0f;
    for (const auto& p : quad) {
        if (p.x <= tol || p.y <= tol ||
            p.x >= static_cast<float>(imgSize.width - 1) - tol ||
            p.y >= static_cast<float>(imgSize.height - 1) - tol) {
            ++borderTouches;
        }
    }
    return borderTouches <= 1;
}

/**
 * @brief Extrait un quadrilatère convexe dominant à partir d'une liste de contours.
 */
std::vector<cv::Point2f> extractBestQuadrilateral(const std::vector<std::vector<cv::Point>>& contours,
                                                  double minRequiredArea,
                                                  const cv::Size& imgSize) {
    std::vector<cv::Point2f> fallbackVertices;
    
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < minRequiredArea) break;
        if (area > (imgSize.width * imgSize.height) * 0.92) continue;
        
        std::vector<cv::Point> hull;
        cv::convexHull(contour, hull);
        double perimeter = cv::arcLength(hull, true);
        
        for (double epsFactor = 0.015; epsFactor <= 0.075; epsFactor += 0.005) {
            std::vector<cv::Point> poly;
            cv::approxPolyDP(hull, poly, epsFactor * perimeter, true);
            if (poly.size() == 4 && cv::isContourConvex(poly)) {
                std::vector<cv::Point2f> candidate;
                for (const auto& pt : poly) {
                    candidate.emplace_back(static_cast<float>(pt.x), static_cast<float>(pt.y));
                }
                candidate = sortQuadrilateralVertices(candidate);
                if (isInteriorDocumentQuad(candidate, imgSize)) {
                    return candidate;
                }
                if (fallbackVertices.empty()) {
                    fallbackVertices = candidate;
                }
            }
        }
    }
    return fallbackVertices;
}

/**
 * @brief MÉTHODE 1 : Détection pour images à FOND SIMPLE (uniforme et contrasté).
 *        Justification : La distribution bimodale des intensités permet au
 *        seuillage global d'Otsu (ou par saturation HSV lorsque la table est claire
 *        mais colorée) de séparer optimalement la page blanche du fond.
 *        Une fermeture morphologique unifie l'intérieur de la page (texte/encre).
 */
std::vector<cv::Point2f> detectOnSimpleBackground(const cv::Mat& image) {
    cv::Mat gray, smoothed, binary;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    
    // Lissage Gaussien pour éliminer le bruit capteur haute fréquence
    cv::GaussianBlur(gray, smoothed, cv::Size(5, 5), 0);
    
    // Binarisation statistique d'Otsu maximisant la variance inter-classes
    cv::threshold(smoothed, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    
    // Fermeture morphologique pour combler le texte imprimé sur la feuille
    cv::Mat structElem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, structElem);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    std::sort(contours.begin(), contours.end(), [](const auto& c1, const auto& c2) {
        return cv::contourArea(c1) > cv::contourArea(c2);
    });
    
    double areaThreshold = (image.cols * image.rows) * 0.08;
    std::vector<cv::Point2f> quad = extractBestQuadrilateral(contours, areaThreshold, image.size());
    
    // Complément colorimétrique multi-canal :
    // 1) Isolation de la feuille blanche faiblement saturée sur table claire teintée (ex: pa.JPG)
    // 2) Isolation de documents comportant des bandeaux colorés sombres ET blancs (ex: dépliant pa_.jpeg)
    cv::Mat hsv, satMask, valMask, whitePageMask;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsv, hsvChannels);
    
    cv::GaussianBlur(hsvChannels[1], hsvChannels[1], cv::Size(7, 7), 0);
    cv::threshold(hsvChannels[1], satMask, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    cv::threshold(hsvChannels[2], valMask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::bitwise_and(satMask, valMask, whitePageMask);
    cv::morphologyEx(whitePageMask, whitePageMask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15)));
    
    contours.clear();
    cv::findContours(whitePageMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::sort(contours.begin(), contours.end(), [](const auto& c1, const auto& c2) {
        return cv::contourArea(c1) > cv::contourArea(c2);
    });
    
    std::vector<cv::Point2f> hsvQuad = extractBestQuadrilateral(contours, areaThreshold, image.size());
    
    // 3) Différence chromatique Rouge - Vert (utile pour un document avec bandeau vert posé sur table marron)
    std::vector<cv::Mat> bgrChannels;
    cv::split(image, bgrChannels);
    cv::Mat rgDiff, bgInvMask;
    cv::subtract(bgrChannels[2], bgrChannels[1], rgDiff);
    cv::GaussianBlur(rgDiff, rgDiff, cv::Size(7, 7), 0);
    cv::threshold(rgDiff, bgInvMask, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    cv::morphologyEx(bgInvMask, bgInvMask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15)));
    
    std::vector<std::vector<cv::Point>> rgContours;
    cv::findContours(bgInvMask, rgContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::sort(rgContours.begin(), rgContours.end(), [](const auto& c1, const auto& c2) {
        return cv::contourArea(c1) > cv::contourArea(c2);
    });
    std::vector<cv::Point2f> rgQuad = extractBestQuadrilateral(rgContours, areaThreshold, image.size());
    
    auto validArea = [&](const std::vector<cv::Point2f>& q) {
        if (q.size() != 4 || !isInteriorDocumentQuad(q, image.size())) return 0.0;
        double a = cv::contourArea(q);
        return (a < (image.cols * image.rows) * 0.85) ? a : 0.0;
    };
    
    double aOtsu = validArea(quad);
    double aHsv  = validArea(hsvQuad);
    double aRg   = validArea(rgQuad);
    
    if (aRg > aOtsu * 1.15 && aRg > aHsv * 1.15) return rgQuad;
    if (aOtsu > 0.0) return quad;
    if (aHsv > 0.0) return hsvQuad;
    
    return (hsvQuad.size() == 4) ? hsvQuad : quad;
}

/**
 * @brief MÉTHODE 2 : Détection pour images à FOND TEXTURÉ (complexe ou bruité).
 *        Justification : Sur un fond texturé (veines de bois, tissu, bruit), un
 *        seuillage direct échoue. Le filtre bilatéral efface les textures internes
 *        de faible amplitude tout en préservant les bords francs de la feuille.
 *        Le gradient morphologique et le détecteur de Canny isolent ensuite la
 *        structure géométrique externe du document.
 */
std::vector<cv::Point2f> detectOnTexturedBackground(const cv::Mat& image) {
    cv::Mat gray, edgePreserved, morphGrad, binaryGrad;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    
    // 1. Filtrage bilatéral : homogénéise les textures tout en conservant les arêtes de la page
    cv::bilateralFilter(gray, edgePreserved, 9, 75, 75);
    
    // 2. Gradient morphologique + Otsu + Fermeture large
    cv::Mat gradElem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(edgePreserved, morphGrad, cv::MORPH_GRADIENT, gradElem);
    cv::threshold(morphGrad, binaryGrad, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    
    cv::Mat closeElem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    cv::morphologyEx(binaryGrad, binaryGrad, cv::MORPH_CLOSE, closeElem);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryGrad, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::sort(contours.begin(), contours.end(), [](const auto& c1, const auto& c2) {
        return cv::contourArea(c1) > cv::contourArea(c2);
    });
    
    double areaThreshold = (image.cols * image.rows) * 0.08;
    std::vector<cv::Point2f> quad = extractBestQuadrilateral(contours, areaThreshold, image.size());
    if (quad.size() == 4 && isInteriorDocumentQuad(quad, image.size())) return quad;
    
    // 3. Repli multi-échelle par détecteur d'arêtes de Canny avec hystérésis
    cv::Mat cannyEdges;
    cv::Canny(edgePreserved, cannyEdges, 35, 125);
    cv::dilate(cannyEdges, cannyEdges, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
    
    contours.clear();
    cv::findContours(cannyEdges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    std::sort(contours.begin(), contours.end(), [](const auto& c1, const auto& c2) {
        return cv::contourArea(c1) > cv::contourArea(c2);
    });
    
    std::vector<cv::Point2f> cannyQuad = extractBestQuadrilateral(contours, areaThreshold, image.size());
    return (cannyQuad.size() == 4) ? cannyQuad : quad;
}

/**
 * @brief Orchestre la détection automatique des 4 coins selon le mode choisi.
 */
std::vector<cv::Point2f> runCornerDetectionPipeline(const cv::Mat& image,
                                                    BackgroundType requestedMode,
                                                    std::string& methodDescription,
                                                    double& complexityMetric) {
    complexityMetric = evaluateBackgroundComplexity(image);
    BackgroundType effectiveMode = requestedMode;
    
    if (requestedMode == BackgroundType::AUTO_DETECT) {
        effectiveMode = (complexityMetric > 95.0) ? BackgroundType::TEXTURED_NOISY
                                                  : BackgroundType::SIMPLE_UNIFORM;
    }
    
    std::vector<cv::Point2f> vertices;
    if (effectiveMode == BackgroundType::SIMPLE_UNIFORM) {
        methodDescription = "Fond Simple (Otsu + Fermeture Morphologique)";
        vertices = detectOnSimpleBackground(image);
        if (vertices.size() != 4 && requestedMode == BackgroundType::AUTO_DETECT) {
            methodDescription = "Fond Simple -> Repli Fond Texture (Bilateral + Gradient)";
            vertices = detectOnTexturedBackground(image);
        }
    } else {
        methodDescription = "Fond Texture (Filtre Bilateral + Gradient Morphologique)";
        vertices = detectOnTexturedBackground(image);
        if (vertices.size() != 4 && requestedMode == BackgroundType::AUTO_DETECT) {
            methodDescription = "Fond Texture -> Repli Fond Simple (Otsu)";
            vertices = detectOnSimpleBackground(image);
        }
    }
    
    return vertices;
}

// ============================================================================
// 4. RENDU GRAPHIQUE ET AJUSTEMENT INTERACTIF (SOURIS & CLAVIER)
// ============================================================================

int findNearestVertex(const std::vector<cv::Point2f>& pts, float cursorX, float cursorY, float radiusLimit = 35.0f) {
    int bestIndex = -1;
    float bestDist = radiusLimit;
    for (size_t i = 0; i < pts.size(); ++i) {
        float dist = std::hypot(pts[i].x - cursorX, pts[i].y - cursorY);
        if (dist < bestDist) {
            bestDist = dist;
            bestIndex = static_cast<int>(i);
        }
    }
    return bestIndex;
}

std::string exportRectifiedDocument(const DocumentScannerContext& ctx, int exportNumber = 0) {
    if (ctx.rectifiedImg.empty()) return "";
    
    std::string outPath;
    if (exportNumber <= 0) {
        outPath = ctx.fileStem + "_redresse" + ctx.fileExt;
    } else {
        outPath = ctx.fileStem + "_redresse_" + std::to_string(exportNumber) + ctx.fileExt;
    }
    
    cv::imwrite(outPath, ctx.rectifiedImg);
    cv::imwrite("document_redresse.jpg", ctx.rectifiedImg);
    return outPath;
}

void refreshVisualInterface(DocumentScannerContext& ctx, bool recomputePerspective = true) {
    ctx.sourceImg.copyTo(ctx.canvasImg);
    
    // Tracé du quadrilatère délimitant la page détectée
    if (ctx.corners.size() == 4) {
        cv::Mat highlightLayer = ctx.canvasImg.clone();
        std::vector<cv::Point> polygon;
        for (const auto& pt : ctx.corners) {
            polygon.emplace_back(static_cast<int>(pt.x), static_cast<int>(pt.y));
        }
        std::vector<std::vector<cv::Point>> polyList = { polygon };
        cv::fillPoly(highlightLayer, polyList, cv::Scalar(40, 220, 100));
        cv::addWeighted(highlightLayer, 0.22, ctx.canvasImg, 0.78, 0, ctx.canvasImg);
        
        for (size_t i = 0; i < 4; ++i) {
            cv::line(ctx.canvasImg, ctx.corners[i], ctx.corners[(i + 1) % 4],
                     cv::Scalar(0, 230, 60), 3, cv::LINE_AA);
        }
    } else {
        for (size_t i = 1; i < ctx.corners.size(); ++i) {
            cv::line(ctx.canvasImg, ctx.corners[i - 1], ctx.corners[i],
                     cv::Scalar(0, 230, 60), 2, cv::LINE_AA);
        }
    }
    
    // Bandeau supérieur d'état
    cv::rectangle(ctx.canvasImg, cv::Point(0, 0), cv::Point(ctx.canvasImg.cols, 54),
                  cv::Scalar(20, 20, 20), -1);
    
    std::string headerLine = "Auteur: Mohammed SALHI | Image: " + ctx.sourceFilePath.filename().string() +
                             " | Methode: " + ctx.activeMethodLabel +
                             " (Score texture: " + std::to_string(static_cast<int>(ctx.borderComplexityMetric)) + ")";
    cv::putText(ctx.canvasImg, headerLine, cv::Point(15, 22),
                cv::FONT_HERSHEY_SIMPLEX, 0.56, cv::Scalar(80, 230, 255), 1, cv::LINE_AA);
                
    std::string helpLine = "Souris: Glisser-deposer les 4 coins | [1] Fond Simple | [2] Fond Texture | [A] Auto | [S] Sauver | [R] Reset | [Q] Quitter";
    cv::putText(ctx.canvasImg, helpLine, cv::Point(15, 44),
                cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(215, 215, 215), 1, cv::LINE_AA);
    
    // Affichage des poignées interactives sur chaque sommet
    const char* cornerNames[4] = {"HG", "HD", "BD", "BG"};
    for (size_t i = 0; i < ctx.corners.size(); ++i) {
        cv::Point2f pt = ctx.corners[i];
        int idx = static_cast<int>(i);
        
        bool dragging = (idx == ctx.activeVertexIdx);
        bool hovering = (idx == ctx.hoveredVertexIdx);
        
        int handleRadius = (dragging || hovering) ? 12 : 8;
        cv::Scalar handleColor = dragging ? cv::Scalar(0, 210, 255)
                                          : (hovering ? cv::Scalar(255, 190, 0)
                                                      : cv::Scalar(20, 40, 245));
        
        cv::circle(ctx.canvasImg, pt, handleRadius, handleColor, -1, cv::LINE_AA);
        cv::circle(ctx.canvasImg, pt, handleRadius + 2, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        
        std::string tag = "C" + std::to_string(i + 1);
        if (i < 4) tag += std::string(" (") + cornerNames[i] + ")";
        if (dragging) {
            tag += " [" + std::to_string(static_cast<int>(pt.x)) + "," + std::to_string(static_cast<int>(pt.y)) + "]";
        }
        cv::putText(ctx.canvasImg, tag,
                    cv::Point(static_cast<int>(pt.x) + 14, static_cast<int>(pt.y) - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    }
    
    cv::imshow(ctx.mainWinTitle, ctx.canvasImg);
    
    if (recomputePerspective && ctx.corners.size() == 4) {
        cv::Mat topDown = computeTopDownPerspective(ctx.sourceImg, ctx.corners);
        if (!topDown.empty()) {
            ctx.rectifiedImg = topDown;
            cv::namedWindow(ctx.previewWinTitle, cv::WINDOW_NORMAL);
            int previewW = std::min(900, ctx.rectifiedImg.cols);
            int previewH = static_cast<int>(static_cast<double>(previewW) * ctx.rectifiedImg.rows / ctx.rectifiedImg.cols);
            cv::resizeWindow(ctx.previewWinTitle, previewW, previewH);
            cv::imshow(ctx.previewWinTitle, ctx.rectifiedImg);
        }
    }
}

void handleMouseInteraction(int event, int x, int y, int /*flags*/, void* userPtr) {
    auto* ctx = reinterpret_cast<DocumentScannerContext*>(userPtr);
    if (!ctx) return;
    
    x = std::clamp(x, 0, ctx->sourceImg.cols - 1);
    y = std::clamp(y, 0, ctx->sourceImg.rows - 1);
    cv::Point2f mousePt(static_cast<float>(x), static_cast<float>(y));
    
    if (event == cv::EVENT_LBUTTONDOWN) {
        if (ctx->corners.size() == 4) {
            int hitIdx = findNearestVertex(ctx->corners, mousePt.x, mousePt.y, 45.0f);
            if (hitIdx != -1) {
                ctx->activeVertexIdx = hitIdx;
                std::cout << "[Interaction] Coin C" << (hitIdx + 1) << " saisi pour ajustement." << std::endl;
            } else {
                int fallbackIdx = findNearestVertex(ctx->corners, mousePt.x, mousePt.y, 260.0f);
                if (fallbackIdx != -1) {
                    ctx->corners[fallbackIdx] = mousePt;
                    ctx->activeVertexIdx = fallbackIdx;
                    std::cout << "[Interaction] Coin C" << (fallbackIdx + 1)
                              << " repositionne vers (" << x << ", " << y << ")." << std::endl;
                }
            }
        } else {
            ctx->corners.push_back(mousePt);
            std::cout << "[Selection manuelle] Point C" << ctx->corners.size()
                      << " ajoute en (" << x << ", " << y << ")." << std::endl;
            if (ctx->corners.size() == 4) {
                ctx->corners = sortQuadrilateralVertices(ctx->corners);
            }
        }
        refreshVisualInterface(*ctx, true);
    }
    else if (event == cv::EVENT_MOUSEMOVE) {
        if (ctx->activeVertexIdx != -1) {
            ctx->corners[ctx->activeVertexIdx] = mousePt;
            refreshVisualInterface(*ctx, true);
        } else if (ctx->corners.size() == 4) {
            int hoverIdx = findNearestVertex(ctx->corners, mousePt.x, mousePt.y, 35.0f);
            if (hoverIdx != ctx->hoveredVertexIdx) {
                ctx->hoveredVertexIdx = hoverIdx;
                refreshVisualInterface(*ctx, false);
            }
        }
    }
    else if (event == cv::EVENT_LBUTTONUP) {
        if (ctx->activeVertexIdx != -1) {
            std::cout << "[Validation] Coin C" << (ctx->activeVertexIdx + 1)
                      << " valide a la position ("
                      << static_cast<int>(ctx->corners[ctx->activeVertexIdx].x) << ", "
                      << static_cast<int>(ctx->corners[ctx->activeVertexIdx].y) << ")." << std::endl;
            ctx->activeVertexIdx = -1;
            ctx->corners = sortQuadrilateralVertices(ctx->corners);
            refreshVisualInterface(*ctx, true);
            
            std::string out = exportRectifiedDocument(*ctx);
            if (!out.empty()) {
                std::cout << "[Sauvegarde] Vue de dessus mise a jour : " << out << std::endl;
            }
        }
    }
}

bool launchAutomaticDetection(DocumentScannerContext& ctx, BackgroundType mode) {
    ctx.selectedMode = mode;
    ctx.activeVertexIdx = -1;
    ctx.hoveredVertexIdx = -1;
    
    std::cout << "\n--------------------------------------------------------------------\n"
              << "[Etape 1] Lancement de la detection automatique des 4 coins...\n";
    auto detectedCorners = runCornerDetectionPipeline(
        ctx.sourceImg, mode, ctx.activeMethodLabel, ctx.borderComplexityMetric);
    
    std::cout << "  -> Indice de texture peripherique : " << std::fixed << std::setprecision(2)
              << ctx.borderComplexityMetric << "\n"
              << "  -> Methode selectionnee           : " << ctx.activeMethodLabel << std::endl;
    
    if (detectedCorners.size() == 4) {
        ctx.corners = detectedCorners;
        std::cout << "[Etape 2] Quadrilatere detecte avec succes :\n";
        const char* labels[4] = {"Haut-Gauche", "Haut-Droit ", "Bas-Droit  ", "Bas-Gauche "};
        for (size_t i = 0; i < 4; ++i) {
            std::cout << "  - Coin C" << (i + 1) << " (" << labels[i] << ") : ("
                      << static_cast<int>(ctx.corners[i].x) << ", "
                      << static_cast<int>(ctx.corners[i].y) << ")\n";
        }
        refreshVisualInterface(ctx, true);
        std::string savedFile = exportRectifiedDocument(ctx);
        if (!savedFile.empty()) {
            std::cout << "[Etape 4] Transformation de perspective appliquee -> " << savedFile << "\n"
                      << "--------------------------------------------------------------------" << std::endl;
        }
        return true;
    } else {
        std::cout << "[Avertissement] Detection automatique partielle. Veuillez pointer les 4 coins a la souris.\n"
                  << "--------------------------------------------------------------------" << std::endl;
        refreshVisualInterface(ctx, false);
        return false;
    }
}

void clearManualSelection(DocumentScannerContext& ctx) {
    ctx.corners.clear();
    ctx.activeVertexIdx = -1;
    ctx.hoveredVertexIdx = -1;
    ctx.sourceImg.copyTo(ctx.canvasImg);
    cv::imshow(ctx.mainWinTitle, ctx.canvasImg);
    try {
        cv::destroyWindow(ctx.previewWinTitle);
    } catch (...) {}
    std::cout << "[Reinitialisation] Coins effaces. Cliquez sur les 4 sommets du document." << std::endl;
}

// ============================================================================
// 5. POINT D'ENTRÉE PRINCIPAL
// ============================================================================

int main(int argc, char** argv) {
    std::string targetArg = "";
    BackgroundType startMode = BackgroundType::AUTO_DETECT;
    
    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        if (flag == "--simple" || flag == "-s") {
            startMode = BackgroundType::SIMPLE_UNIFORM;
        } else if (flag == "--textured" || flag == "-t") {
            startMode = BackgroundType::TEXTURED_NOISY;
        } else if (flag == "--auto" || flag == "-a") {
            startMode = BackgroundType::AUTO_DETECT;
        } else if (flag == "--help" || flag == "-h") {
            std::cout << "Auteur : Mohammed SALHI\n"
                      << "Usage  : " << argv[0] << " [chemin_image] [--simple|--textured|--auto]\n";
            return 0;
        } else if (!flag.empty() && flag[0] != '-') {
            targetArg = flag;
        }
    }
    
    std::string resolvedPath = resolveTargetImage(targetArg);
    fs::path pathObj(resolvedPath);
    
    std::cout << "====================================================================\n"
              << "  REDRESSEMENT DE PERSPECTIVE DE DOCUMENTS - OpenCV C++\n"
              << "  Realise par : Mohammed SALHI\n"
              << "====================================================================\n"
              << "Image source   : " << resolvedPath << "\n"
              << "Commandes interactives :\n"
              << "  * Souris (Glisser-Deposer) : Ajuster interactivement les 4 coins\n"
              << "  * Touche '1'               : Mode Fond Simple (Otsu + Morphologie)\n"
              << "  * Touche '2'               : Mode Fond Texture (Bilateral + Gradient)\n"
              << "  * Touche 'a' / 'A'         : Mode Auto-adaptatif\n"
              << "  * Touche 'r' / 'R'         : Reinitialiser et pointer manuellement\n"
              << "  * Touche 's' / 'S'         : Exporter l'image orthorectifiee\n"
              << "  * Touche 'q' / Echap       : Quitter le programme\n"
              << "====================================================================\n";
              
    DocumentScannerContext ctx;
    ctx.sourceFilePath = pathObj;
    ctx.fileStem = pathObj.stem().string();
    ctx.fileExt = pathObj.has_extension() ? normalizeExt(pathObj.extension().string()) : ".jpg";
    
    ctx.sourceImg = cv::imread(resolvedPath, cv::IMREAD_COLOR);
    if (ctx.sourceImg.empty()) {
        std::cerr << "[Erreur] Impossible de charger le fichier image : " << resolvedPath << std::endl;
        return -1;
    }
    
    std::cout << "[Chargement] Resolution : " << ctx.sourceImg.cols << " x "
              << ctx.sourceImg.rows << " pixels." << std::endl;
              
    ctx.sourceImg.copyTo(ctx.canvasImg);
    
    cv::namedWindow(ctx.mainWinTitle, cv::WINDOW_NORMAL);
    int displayW = std::min(1200, ctx.sourceImg.cols);
    int displayH = static_cast<int>(static_cast<double>(displayW) * ctx.sourceImg.rows / ctx.sourceImg.cols);
    cv::resizeWindow(ctx.mainWinTitle, displayW, displayH);
    
    cv::setMouseCallback(ctx.mainWinTitle, handleMouseInteraction, &ctx);
    cv::imshow(ctx.mainWinTitle, ctx.canvasImg);
    
    // Exécution initiale de la détection automatique et de la transformation
    launchAutomaticDetection(ctx, startMode);
    
    int manualSaveIdx = 1;
    while (true) {
        char key = static_cast<char>(cv::waitKey(20));
        if (key == 'q' || key == 'Q' || key == 27) {
            std::cout << "Arret du programme." << std::endl;
            break;
        }
        if (key == '1') {
            launchAutomaticDetection(ctx, BackgroundType::SIMPLE_UNIFORM);
        } else if (key == '2') {
            launchAutomaticDetection(ctx, BackgroundType::TEXTURED_NOISY);
        } else if (key == 'a' || key == 'A') {
            launchAutomaticDetection(ctx, BackgroundType::AUTO_DETECT);
        } else if (key == 'r' || key == 'R') {
            clearManualSelection(ctx);
        } else if (key == 's' || key == 'S') {
            std::string out = exportRectifiedDocument(ctx, manualSaveIdx++);
            if (!out.empty()) {
                std::cout << "[Export] Document redresse enregistre sous : " << out << std::endl;
            }
        }
    }
    
    cv::destroyAllWindows();
    return 0;
}
