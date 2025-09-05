!(import! &db data)

!(bind! &db2 (new-space))

(= (foldlhelper $x $acc)
    (let*
        (
            (($a $b $dur $cost $dist) $x)
            (($c $d $first $second $third) $acc)
        )
        ((cons-atom $a $c) (cons-atom $b $d) (cons-atom $dur $first) (cons-atom $cost $second) (cons-atom $dist $third))
    )
)

(= (builder ($cityA $cityB $dur $cost $dist) $alldur $allcosts $alldist)
    (let*
        (
            ($maxdur (max-atom $alldur))
            ($mindur (min-atom $alldur))    
            ($newdur (/ (- $dur $mindur) (- $maxdur $mindur)))

            ($maxcost (max-atom $allcosts))
            ($mincost (min-atom $allcosts))
            ($newcost (/ (- $cost $mincost) (- $maxcost $mincost)))
            
            ($maxdist (max-atom $alldist))
            ($mindist (min-atom $alldist))
            ($newdist (/ (- $dist $mindist) (- $maxdist $mindist)))  

            ($overall (+ (+ $newdist $newdur) $newcost))      

            ; (() (println! (flight-route $cityA $cityB (Duration $newdur Cost $newcost Distance $newdist Overall $overall))))
        )
        (flight-route $cityA $cityB (Duration $dur Cost $cost Distance $dist Overall $overall))
    )
)
(= (dataPreProcessor)
    (let*
        (
            ($expr (collapse (match &db (flight-route $x $y (Duration $z Cost $w Distance $u)) ($x $y $z $w $u))))
            (($allcityA $allcityB $alldur $allcosts $alldist) (foldl-atom $expr (() () () () ()) $acc $x (foldlhelper $x $acc)))
        )
        (add-reduct &db2 (let $each (superpose $expr) (builder $each $alldur $allcosts $alldist)))
    )
)

(= (isMemberHelper $list $ele) 
    (collapse (let $child (superpose $list) (if (== $child $ele) True (empty))))
)

(= (isMember $list $ele)
    (if (== (isMemberHelper $list $ele) (True)) True False)
)

; (: pathFind (-> Atom Atom Expression Number SpaceAtom Expression))
(= (pathFind $first $last $visited $type $space)
    (if (isMember $visited $first) 
        (empty)
        (if (== $first $last) 
            ($first)
            (let $newVisited (cons-atom $first $visited)
                (match $space 
                    (flight-route $first $sec $remaining)
                    (let ($ans $x) 
                        ((pathFind $sec $last $newVisited $type $space) (index-atom $remaining (+ (* $type 2) 1)))
                        (union-atom ($first -- $x -->) $ans)
                    )
                )
            )
        )
    )
)

; (: pathFind (-> Atom Atom Number SpaceAtom Expression))
(= (pathFind $first $second $type $space)
    (if (== $space 1)
        (collapse (pathFind $first $second () $type &db))
        (let ($_ $answer) ((dataPreProcessor) (collapse (pathFind $first $second () $type &db2))) $answer)
    )
)

; (: pathFind (-> Atom Atom Number Expression))
(= (pathFind $first $second $type)
    (collapse (pathFind $first $second () $type &db))
)


; !(pathFind Jimma Addis 2)


(= (helper $path) 
    (let $y (collapse (let $each (superpose $path) (if (== (get-type $each) Number) $each (empty)))) (foldl-atom $y 0 $acc $x (+ $acc $x)))
)

(= (shortestPathFinderHelper $allpaths $shPath $minimum $index) 
    (if (<= (size-atom $allpaths) $index)
        $shPath
        (let*
            (
                ($path (index-atom $allpaths $index))
                ($current (helper $path))
                ($newMin (min-atom ($minimum $current)))
                ($newShPath (if (< $current $minimum) $path $shPath))
            )
            (shortestPathFinderHelper $allpaths $newShPath $newMin (+ $index 1))
        )
    )
)

(= (shortestPathFinder $first $second $type $space)
    (let $allpaths (pathFind $first $second $type $space)
        (if (== $allpaths ())
            (There is no path between $first and $second)
            (let $shPath (index-atom $allpaths 0) (shortestPathFinderHelper $allpaths $shPath (py-atom "float('inf')") 0))
        )
    )
)

(= (shortestPathFinder $first $second $type)
    (let $allpaths (pathFind $first $second $type 1)
        (if (== $allpaths ())
            (There is no path between $first and $second)
            (let $shPath (index-atom $allpaths 0) (shortestPathFinderHelper $allpaths $shPath (py-atom "float('inf')") 0))
        )
    )
)

!(shortestPathFinder Addis Jimma 1)
